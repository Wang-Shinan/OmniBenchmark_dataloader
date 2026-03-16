"""
ADHD Dataset Builder.

Dataset: ADHD Dataset (likely TUMS / Nasrabadi et al.)
- Task: ADHD vs Control Classification
- Sampling Rate: 128 Hz
- Channels: 19 (10-20 standard)

Data Structure:
- ADHD_part1/
  - v1p.mat, v2p.mat...
- Control_part1/
  - v1.mat, v2.mat...

Processing:
1. Load .mat files
2. Transpose to (channels, time)
3. Standardize channel names to 10-20 system (Uppercase)
4. Segment into 2s windows (no overlap by default, or with stride)
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from ..utils import ElectrodeSet
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from utils import ElectrodeSet

# Standard 10-20 channels in the order appearing in .ced file / data
ADHD_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]

ADHD_INFO = DatasetInfo(
    dataset_name="ADHD_TUMS",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["Control", "ADHD"],
    sampling_rate=128.0,
    montage="10_20",
    channels=ADHD_CHANNELS
)

DEFAULT_MAX_AMPLITUDE_UV = 800.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    max_amp = np.abs(data).max()
    if max_amp > 1e-3:  # likely µV
        return data / 1e6, "µV"
    return data, "V"


class ADHDBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 60.0, ## need lower than original 64Hz Nyquist
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "ADHD"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 128.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Verify electrode names
        self.electrode_set = ElectrodeSet()
        self._verify_channels()

        self.files_map = self._scan_files()

    def _verify_channels(self):
        """Ensure all channels are valid according to standard schema."""
        for ch in ADHD_CHANNELS:
            if not self.electrode_set.is_valid_electrode(ch):
                print(f"Warning: Channel {ch} is not in standard ElectrodeSet!")

    def _scan_files(self):
        """Scan directories and map subject ID to file info."""
        files = []

        # ADHD (Label 1)
        for part in ["ADHD_part1", "ADHD_part2"]:
            p = self.raw_data_dir / part
            if p.exists():
                for f in p.glob("*.mat"):
                    files.append({
                        "path": f,
                        "label": 1,
                        "group": "ADHD"
                    })
            else:
                print(f"Warning: Directory {p} does not exist.")

        # Control (Label 0)
        for part in ["Control_part1", "Control_part2"]:
            p = self.raw_data_dir / part
            if p.exists():
                for f in p.glob("*.mat"):
                    files.append({
                        "path": f,
                        "label": 0,
                        "group": "Control"
                    })
            else:
                print(f"Warning: Directory {p} does not exist.")

        # Sort by filename to ensure consistency
        files.sort(key=lambda x: x["path"].name)

        if not files:
            raise FileNotFoundError(f"No .mat files found in {self.raw_data_dir}")

        return {i: f for i, f in enumerate(files)}

    def get_subject_ids(self) -> list[int]:
        return list(self.files_map.keys())

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_trials}")
        print(f"  Valid segments: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_trials} ({100 - valid_pct:.1f}%)")

    def _convert_to_mne(self, data: np.ndarray):
        """Convert numpy array to MNE Raw object."""
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        print(f"  Detected unit: {detected_unit}, max amplitude: {np.abs(data).max():.2e}")

        info = mne.create_info(
            ch_names=ADHD_CHANNELS,
            sfreq=self.orig_sfreq,
            ch_types=['eeg'] * len(ADHD_CHANNELS)
        )
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": ADHD_INFO.dataset_name,
                "task_type": str(ADHD_INFO.task_type.value),
                "downstream_task": str(ADHD_INFO.downstream_task_type.value),
                "num_labels": ADHD_INFO.num_labels,
                "category_list": ADHD_INFO.category_list,
                "original_sampling_rate": ADHD_INFO.sampling_rate,
                "channels": ADHD_INFO.channels,
                "montage": ADHD_INFO.montage,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_subject(self, subject_id: int) -> str:
        if not HAS_MNE:
            raise ImportError("MNE is required to build this dataset")

        if subject_id not in self.files_map:
            raise ValueError(f"Subject ID {subject_id} not found.")

        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        info = self.files_map[subject_id]
        file_path = info["path"]
        label = info["label"]

        print(f"Processing Subject {subject_id}: {file_path.name} (Label: {label})")

        mat = scipy.io.loadmat(file_path)
        key = file_path.stem
        if key not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if len(keys) == 1:
                key = keys[0]
            else:
                largest_key = max(keys, key=lambda k: mat[k].size)
                key = largest_key
                print(f"  Note: Using key '{key}' for {file_path.name}")

        data = mat[key]

        if data.shape[1] == 19:
            eeg_data = data.T
        elif data.shape[0] == 19:
            eeg_data = data
        else:
            raise ValueError(f"Unexpected shape {data.shape} for {file_path.name}. Expected 19 channels.")

        raw = self._convert_to_mne(eeg_data)
        raw = self._preprocess(raw)

        eeg_data_uv = raw.get_data() * 1e6
        ch_names = raw.ch_names

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=ADHD_INFO.dataset_name,
            task_type=ADHD_INFO.task_type.value,
            downstream_task_type=ADHD_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=ADHD_INFO.num_labels,
            category_list=ADHD_INFO.category_list,
            chn_type="EEG",
            montage=ADHD_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_file = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(out_file), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=0, session_id=0)
            trial_name = writer.add_trial(trial_attrs)

            n_samples = eeg_data_uv.shape[1]

            for i_slice, start in enumerate(range(0, n_samples - self.window_samples + 1, self.stride_samples)):
                end = start + self.window_samples
                seg_data = eeg_data_uv[:, start:end]

                self.total_trials += 1
                if not self._validate_trial(seg_data):
                    self.rejected_trials += 1
                    continue
                self.valid_trials += 1

                seg_attrs = SegmentAttrs(
                    segment_id=i_slice,
                    start_time=start / self.target_sfreq,
                    end_time=end / self.target_sfreq,
                    time_length=self.window_sec,
                    label=np.array([label])
                )
                writer.add_segment(trial_name, seg_attrs, seg_data)

        self._report_validation_stats(subject_id)
        print(f"Saved {out_file}")
        return str(out_file)

    def build_all(self, subject_ids: list[int] = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total = 0
        all_valid = 0
        all_rejected = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total += self.total_trials
                all_valid += self.valid_trials
                all_rejected += self.rejected_trials
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)

        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal segments: {all_total}")
        print(f"Valid segments: {all_valid}")
        print(f"Rejected segments: {all_rejected}")
        if all_total > 0:
            print(f"Rejection rate: {all_rejected / all_total * 100:.1f}%")
        print("=" * 50)

        stats = {
            "total_subjects": len(subject_ids),
            "successful": len(output_paths),
            "failed": len(failed_subjects),
            "failed_subject_ids": failed_subjects,
            "total_segments": all_total,
            "valid_segments": all_valid,
            "rejected_segments": all_rejected,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_adhd(
    raw_data_dir: str,
    output_dir: str = "/mnt/dataset2/Processed_datasets/EEG_Bench",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build ADHD dataset."""
    builder = ADHDBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build ADHD HDF5 dataset")
    parser.add_argument("data_dir", type=str, default="/mnt/dataset2/Datasets/ADHD", help="Directory containing raw files")
    parser.add_argument("--output_dir", type=str, default="/mnt/dataset2/Processed_datasets/EEG_Bench", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_adhd(args.data_dir, args.output_dir, args.subjects)
