"""
EEG-IO_4Class Dataset Builder.

EEG-IO: Involuntary eye blinks (stimulus-driven) from OpenBCI.
- 20 subjects
- 4 classes: normal blink, stimulus blink, soft blink, no blink
- link to dataset: https://gnan.ece.gatech.edu/eeg-eyeblinks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, cast
import csv
import json
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

if TYPE_CHECKING:
    import mne as mne_types

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


EEGIO_INFO = DatasetInfo(
    dataset_name="EEG-IO_4Class",
    task_type=DatasetTaskType.EOG,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=4,
    category_list=["normal blink", "stimulus blink", "soft blink", "no blink"],
    sampling_rate=250.0,
    montage="10_20",
    channels=["Fp1", "Fp2"],
)

REMOVE_CHANNELS: list[str] = []

DEFAULT_MAX_AMPLITUDE_UV = 600.0
DEFAULT_ROBUST_CLAMP_UV = 400.0


@dataclass(frozen=True)
class Segment:
    data: np.ndarray
    label: int
    start_time: float
    end_time: float


class EEGIOBuilder:
    """Builder for EEG-IO dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.5,
        filter_high: float = 40.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        skip_corrupt: bool = True,
        robust_clamp_uv: float = DEFAULT_ROBUST_CLAMP_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "EEG-IO"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = EEGIO_INFO.sampling_rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.skip_corrupt = skip_corrupt
        self.robust_clamp_uv = robust_clamp_uv

        self.window_samples = int(round(window_sec * target_sfreq))
        self.stride_samples = int(round(stride_sec * target_sfreq))

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0
        self.last_label_counts: dict[int, int] | None = None

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (0-19)."""
        return list(range(0, 20))

    def _find_files(self, subject_id: int) -> tuple[Path, Path]:
        """Find data and label files for a subject."""
        sid_str = f"{subject_id:02d}"
        data_path = self.raw_data_dir / f"S{sid_str}_data.csv"
        label_path = self.raw_data_dir / f"S{sid_str}_labels.csv"
        if not data_path.is_file():
            raise FileNotFoundError(f"EEG data file not found: {data_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        return data_path, label_path

    def _read_raw(self, file_path: Path):
        """Read raw EEG file and convert to MNE Raw object."""
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")
        data = np.loadtxt(file_path, delimiter=";", skiprows=1, usecols=(1, 2))
        data_volts = data / 1e6
        mne_module = cast("mne_types", mne)
        info = mne_module.create_info(EEGIO_INFO.channels, sfreq=self.orig_sfreq, ch_types="eeg")
        return mne_module.io.RawArray(data_volts.T, info, verbose=False)

    def _read_labels(self, label_path: Path, recording_end: float) -> tuple[list[tuple[float, float]], np.ndarray]:
        """Parse corrupt intervals and blink events from label CSV."""
        corrupt_intervals: list[tuple[float, float]] = []
        blinks: list[tuple[float, int]] = []
        remaining_corrupt = 0
        with open(label_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if not row:
                    continue
                key = row[0].strip().lower()
                if key == "corrupt":
                    remaining_corrupt = int(row[1])
                elif remaining_corrupt > 0:
                    start = float(row[0])
                    end = float(row[1])
                    if end == -1:
                        end = recording_end
                    corrupt_intervals.append((start, end))
                    remaining_corrupt -= 1
                elif key == "blinks":
                    continue
                else:
                    blinks.append((float(row[0]), int(row[1])))
        return corrupt_intervals, np.array(blinks)

    @staticmethod
    def _overlaps_window(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
        for interval_start, interval_end in intervals:
            if max(start, interval_start) < min(end, interval_end):
                return True
        return False

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _robust_clamp(self, trial_data: np.ndarray) -> np.ndarray:
        """Center and scale data using median/P99, then clip."""
        if self.robust_clamp_uv <= 0:
            return trial_data
        centered = trial_data - np.median(trial_data, axis=1, keepdims=True)
        scale_base = np.percentile(np.abs(centered), 99, axis=1, keepdims=True)
        scale = np.where(scale_base == 0, 1.0, self.robust_clamp_uv / scale_base)
        scaled = centered * scale
        return np.clip(scaled, -self.robust_clamp_uv, self.robust_clamp_uv)

    def _report_validation_stats(self, subject_id: int) -> None:
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} uV")
        print(f"  Total segments: {self.total_trials}")
        print(f"  Valid segments: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def _preprocess(self, raw: "mne_types.io.BaseRaw") -> "mne_types.io.BaseRaw":
        """Apply preprocessing to raw data."""
        if REMOVE_CHANNELS:
            raw.drop_channels(REMOVE_CHANNELS)

        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": EEGIO_INFO.dataset_name,
                "task_type": str(EEGIO_INFO.task_type.value),
                "downstream_task": str(EEGIO_INFO.downstream_task_type.value),
                "num_labels": EEGIO_INFO.num_labels,
                "category_list": EEGIO_INFO.category_list,
                "original_sampling_rate": EEGIO_INFO.sampling_rate,
                "channels": EEGIO_INFO.channels,
                "montage": EEGIO_INFO.montage,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "skip_corrupt": self.skip_corrupt,
                "robust_clamp_uv": self.robust_clamp_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_subject(self, subject_id: int) -> str:
        """Build HDF5 file for a single subject."""
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0
        self.last_label_counts = None

        data_path, label_path = self._find_files(subject_id)

        print(f"Reading {data_path}")
        raw = self._read_raw(data_path)
        raw = self._preprocess(raw)

        data_uv = np.asarray(raw.get_data()) * 1e6
        data_uv = self._robust_clamp(data_uv)
        sfreq = float(raw.info["sfreq"])
        recording_end = data_uv.shape[1] / sfreq

        corrupt_intervals, blinks = self._read_labels(label_path, recording_end)

        segments: list[Segment] = []
        blink_windows: list[tuple[float, float]] = []

        for blink_time, blink_code in blinks:
            if blink_code not in (0, 1, 2):
                continue
            start_time = blink_time - self.window_sec / 2
            end_time = start_time + self.window_sec
            if start_time < 0 or end_time > recording_end:
                continue
            if self.skip_corrupt and self._overlaps_window(start_time, end_time, corrupt_intervals):
                continue
            start_idx = int(round(start_time * sfreq))
            end_idx = start_idx + self.window_samples
            if end_idx > data_uv.shape[1]:
                continue
            segments.append(
                Segment(
                    data=data_uv[:, start_idx:end_idx],
                    label=int(blink_code),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            blink_windows.append((start_time, end_time))

        n_samples = data_uv.shape[1]
        for start_idx in range(0, n_samples - self.window_samples + 1, self.stride_samples):
            start_time = start_idx / sfreq
            end_time = start_time + self.window_sec
            if self._overlaps_window(start_time, end_time, blink_windows):
                continue
            if self.skip_corrupt and self._overlaps_window(start_time, end_time, corrupt_intervals):
                continue
            segments.append(
                Segment(
                    data=data_uv[:, start_idx : start_idx + self.window_samples],
                    label=3,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        if not segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=EEGIO_INFO.dataset_name,
            task_type=EEGIO_INFO.task_type.value,
            downstream_task_type=EEGIO_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=EEGIO_INFO.channels,
            num_labels=EEGIO_INFO.num_labels,
            category_list=EEGIO_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=EEGIO_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"
        label_counts: dict[int, int] = {label: 0 for label in range(EEGIO_INFO.num_labels)}
        self.last_label_counts = label_counts

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=0, session_id=0)
            trial_name = writer.add_trial(trial_attrs)

            segment_id = 0
            for segment in segments:
                self.total_trials += 1
                if not self._validate_trial(segment.data):
                    self.rejected_trials += 1
                    continue
                self.valid_trials += 1
                label_counts[segment.label] = label_counts.get(segment.label, 0) + 1

                segment_attrs = SegmentAttrs(
                    segment_id=segment_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    time_length=self.window_sec,
                    label=np.array([segment.label]),
                )
                writer.add_segment(trial_name, segment_attrs, segment.data)
                segment_id += 1

        self._report_validation_stats(subject_id)
        print(f"Label counts: {label_counts}")
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[int] | None = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0
        all_label_counts: dict[int, int] = {label: 0 for label in range(EEGIO_INFO.num_labels)}
        per_subject_label_counts: dict[int, dict[int, int]] = {}

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
                if self.last_label_counts is not None:
                    per_subject_label_counts[subject_id] = dict(self.last_label_counts)
                    for label, count in self.last_label_counts.items():
                        all_label_counts[label] = all_label_counts.get(label, 0) + count
            except Exception as exc:
                print(f"Error processing subject {subject_id}: {exc}")
                failed_subjects.append(subject_id)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_trials_or_segments": all_total_trials,
            "valid_trials_or_segments": all_valid_trials,
            "rejected_trials_or_segments": all_rejected_trials,
            "label_counts": all_label_counts,
            "label_counts_per_subject": per_subject_label_counts,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_eegio_dataset(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] | None = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build EEG-IO dataset."""
    builder = EEGIOBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build EEG-IO HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_eegio_dataset(args.raw_data_dir, args.output_dir, args.subjects)
