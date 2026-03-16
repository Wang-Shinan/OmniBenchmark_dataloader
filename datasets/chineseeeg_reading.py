"""
ChineseEEG (ds004952) Reading vs Non‑Reading Dataset Builder.

This module implements a preprocessing pipeline for the OpenNeuro dataset
ds004952 "ChineseEEG: A Chinese Linguistic Corpora EEG Dataset for Semantic
Alignment and Neural Decoding", producing a binary classification dataset:

- Label 1: reading (during formal reading rows)
- Label 0: non‑reading (within BEGN–STOP but outside reading rows)

The implementation follows the benchmark_dataloader style using:
- SubjectAttrs / TrialAttrs / SegmentAttrs
- One HDF5 file per subject
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mne

    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    # Fallback for running as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

CHINESEEEG_READING_INFO = DatasetInfo(
    dataset_name="ChineseEEG_reading_binary",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["non_reading", "reading"],
    sampling_rate=256.0,  # target after resampling
    montage="10_20",
    channels=[],
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 800.0


@dataclass
class ReadingIntervals:
    """Holds reading intervals for a single run."""

    # List of (start, end) in seconds, non‑overlapping, sorted
    intervals: List[Tuple[float, float]]
    # Overall valid range [start, end] in seconds (BEGN–STOP or min/max onset)
    global_start: float
    global_end: float


class ChineseEEGReadingBuilder:
    """Builder for ChineseEEG reading vs non‑reading dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 256.0,
        window_sec: float = 2.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.5,
        filter_high: float = 40.0,
        filter_notch: float = 50.0,  # 50 Hz for China
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        reading_min_overlap: float = 0.7,
        nonreading_max_overlap: float = 0.3,
        include_preface_as_reading: bool = True,
    ):
        """
        Initialize ChineseEEG reading vs non‑reading builder.

        Args:
            raw_data_dir: Root of BIDS ds004952 (contains sub-*/).
            output_dir: Output directory for HDF5 files.
            target_sfreq: Target sampling frequency (Hz).
            window_sec: Sliding window length in seconds.
            stride_sec: Sliding window stride in seconds.
            filter_low: Low cutoff for bandpass filter (Hz).
            filter_high: High cutoff for bandpass filter (Hz).
            filter_notch: Notch filter frequency (Hz), 0 to disable.
            max_amplitude_uv: Max allowed window amplitude in µV.
            reading_min_overlap: Minimum proportion of window overlapping
                reading intervals to be labeled as reading.
            nonreading_max_overlap: Maximum reading overlap proportion for
                window to be labeled as non‑reading.
            include_preface_as_reading: If True, treat PRES–PREE as reading.
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == CHINESEEEG_READING_INFO.dataset_name:
            self.output_dir = output_path
        else:
            self.output_dir = output_path / CHINESEEEG_READING_INFO.dataset_name

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.reading_min_overlap = float(reading_min_overlap)
        self.nonreading_max_overlap = float(nonreading_max_overlap)
        self.include_preface_as_reading = bool(include_preface_as_reading)

        self.window_samples = int(self.window_sec * self.target_sfreq)
        self.stride_samples = int(self.stride_sec * self.target_sfreq)

        # Track validation statistics
        self.total_windows = 0
        self.valid_windows = 0
        self.rejected_windows_amp = 0
        self.rejected_windows_ambiguous = 0

        # Dataset‑level channels (first subject/run sets this)
        self._dataset_channels: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # BIDS helpers
    # ------------------------------------------------------------------

    def get_subject_ids(self) -> List[str]:
        """Return list of subject IDs (sub-xx) from BIDS root."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        sub_dirs = sorted(
            p.name for p in self.raw_data_dir.glob("sub-*") if p.is_dir()
        )
        return sub_dirs

    def _find_runs_for_subject(self, subject_id: str) -> List[Dict[str, Path]]:
        """
        Find all BrainVision runs for a subject across sessions.

        Returns list of dicts with keys:
          - 'vhdr': Path to .vhdr
          - 'events': Path to events.tsv (if exists, else None)
          - 'session': session name (e.g., 'ses-LittlePrince')
          - 'run_name': stem of the run file
        """
        sub_dir = self.raw_data_dir / subject_id
        if not sub_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {sub_dir}")

        runs: List[Dict[str, Path]] = []
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue

            for vhdr_file in sorted(eeg_dir.glob("*.vhdr")):
                stem = vhdr_file.stem  # e.g., sub-04_ses-LittlePrince_task-*_eeg
                # BIDS events TSV: same prefix with _events.tsv
                events_tsv = eeg_dir / f"{stem}_events.tsv"
                if not events_tsv.exists():
                    events_tsv = None  # type: ignore[assignment]

                runs.append(
                    {
                        "vhdr": vhdr_file,
                        "events": events_tsv,
                        "session": ses_dir.name,
                        "run_name": stem,
                    }
                )

        return runs

    # ------------------------------------------------------------------
    # Event parsing & reading intervals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_events_tsv(events_tsv: Path) -> pd.DataFrame:
        """Load BIDS events.tsv into a DataFrame with at least onset column."""
        df = pd.read_csv(events_tsv, sep="\t")
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "onset" not in df.columns:
            raise ValueError(f"events.tsv missing 'onset' column: {events_tsv}")

        # Determine event descriptor column
        name_cols = ["trial_type", "value", "event_type"]
        name_col = None
        for c in name_cols:
            if c in df.columns:
                name_col = c
                break
        if name_col is None:
            # Fallback: treat all as unknown
            df["event_name"] = ""
        else:
            df["event_name"] = df[name_col].astype(str)

        return df

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping intervals, return sorted non‑overlapping list."""
        if not intervals:
            return []
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        cur_start, cur_end = intervals_sorted[0]
        for start, end in intervals_sorted[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged

    def _build_reading_intervals_from_events(
        self, events_df: pd.DataFrame
    ) -> ReadingIntervals:
        """
        Build reading intervals based on BIDS events.

        We use markers described in the ds004952 README:
          - BEGN / STOP  : overall recording range
          - ROWS / ROWE  : beginning / end of a reading row (three-line page)
          - PRES / PREE  : preface start / end (optional reading)
        """
        if events_df.empty:
            raise ValueError("Empty events DataFrame")

        # Shorthand
        onset = events_df["onset"].to_numpy(dtype=float)
        names = events_df["event_name"].astype(str).to_numpy()

        # Overall range: BEGN–STOP if available, otherwise min/max onset
        befn_idx = np.where(names == "BEGN")[0]
        stop_idx = np.where(names == "STOP")[0]
        if len(befn_idx) > 0 and len(stop_idx) > 0:
            global_start = float(onset[befn_idx[0]])
            global_end = float(onset[stop_idx[-1]])
        else:
            global_start = float(onset.min())
            global_end = float(onset.max())

        # Collect ROWS–ROWE intervals
        reading_intervals: List[Tuple[float, float]] = []
        current_rows_start: Optional[float] = None

        for t, name in zip(onset, names):
            if name == "ROWS":
                current_rows_start = float(t)
            elif name == "ROWE" and current_rows_start is not None:
                end_t = float(t)
                if end_t > current_rows_start:
                    reading_intervals.append((current_rows_start, end_t))
                current_rows_start = None

        # Optional: PRES–PREE as reading
        if self.include_preface_as_reading:
            pres_idx = np.where(names == "PRES")[0]
            pree_idx = np.where(names == "PREE")[0]
            if len(pres_idx) > 0 and len(pree_idx) > 0:
                pres_t = float(onset[pres_idx[0]])
                pree_t = float(onset[pree_idx[-1]])
                if pree_t > pres_t:
                    reading_intervals.append((pres_t, pree_t))

        # Clip intervals to [global_start, global_end]
        clipped: List[Tuple[float, float]] = []
        for s, e in reading_intervals:
            s_clipped = max(s, global_start)
            e_clipped = min(e, global_end)
            if e_clipped > s_clipped:
                clipped.append((s_clipped, e_clipped))

        merged = self._merge_intervals(clipped)
        return ReadingIntervals(intervals=merged, global_start=global_start, global_end=global_end)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _read_raw(self, vhdr_file: Path):
        """Read BrainVision file and return MNE Raw in Volts."""
        if not HAS_MNE:
            raise ImportError("MNE is required for reading BrainVision files")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_brainvision(
                    str(vhdr_file), preload=True, verbose=False
                )

            # Unit sanity check: MNE usually returns V; if values are extreme, rescale
            data = raw.get_data()
            max_amp = np.abs(data).max()
            if max_amp > 10.0:  # likely µV
                raw._data = raw._data / 1e6  # type: ignore[attr-defined]
                print(f"  Detected unit: µV (max={max_amp:.2e}), converted to V")
            elif max_amp > 1.0:  # likely mV
                raw._data = raw._data / 1e3  # type: ignore[attr-defined]
                print(f"  Detected unit: mV (max={max_amp:.2e}), converted to V")

            return raw
        except Exception as e:  # pragma: no cover - robust error path
            print(f"  Warning: Failed to read {vhdr_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply notch + bandpass filtering and resampling."""
        if self.filter_notch > 0:
            try:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            except Exception:  # pragma: no cover
                pass

        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    # ------------------------------------------------------------------
    # Windowing and labeling
    # ------------------------------------------------------------------

    @staticmethod
    def _interval_overlap(
        a_start: float, a_end: float, b_start: float, b_end: float
    ) -> float:
        """Return overlap length between intervals [a_start,a_end) and [b_start,b_end)."""
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        return max(0.0, end - start)

    def _label_window(
        self,
        t_start: float,
        t_end: float,
        reading_intervals: ReadingIntervals,
    ) -> Optional[int]:
        """
        Label a window as reading (1), non‑reading (0), or None (ambiguous).
        """
        # If window is completely outside global range, skip
        if t_end <= reading_intervals.global_start or t_start >= reading_intervals.global_end:
            return None

        length = t_end - t_start
        if length <= 0:
            return None

        # Compute overlap with reading intervals
        overlap_reading = 0.0
        for rs, re in reading_intervals.intervals:
            if rs >= t_end:
                break
            if re <= t_start:
                continue
            overlap_reading += self._interval_overlap(t_start, t_end, rs, re)

        prop = overlap_reading / length

        if prop >= self.reading_min_overlap:
            return 1  # reading
        if prop <= self.nonreading_max_overlap:
            return 0  # non‑reading

        # Ambiguous: too much mixing of reading and non‑reading
        return None

    def _segment_run(
        self,
        raw,
        reading_intervals: ReadingIntervals,
        session_name: str,
        run_name: str,
        session_id: int,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Segment a run into labeled windows.

        Returns:
            segments: list of dicts with keys:
              - 'data': (n_channels, window_samples) in µV
              - 'label': int (0 or 1)
              - 'start_time', 'end_time'
              - 'session_id', 'run_name'
            ch_names: list of channel names
        """
        raw = self._preprocess(raw)
        ch_names = raw.ch_names
        data_v = raw.get_data()
        sfreq = raw.info["sfreq"]

        # Convert to µV once
        data_uv = data_v * 1e6
        n_channels, n_samples = data_uv.shape

        segments: List[Dict] = []
        trial_counter = 0

        max_start_sample = n_samples - self.window_samples
        if max_start_sample <= 0:
            return segments, ch_names

        for start_sample in range(0, max_start_sample + 1, self.stride_samples):
            end_sample = start_sample + self.window_samples
            if end_sample > n_samples:
                break

            t_start = start_sample / sfreq
            t_end = end_sample / sfreq

            label = self._label_window(t_start, t_end, reading_intervals)
            if label is None:
                self.rejected_windows_ambiguous += 1
                continue

            window = data_uv[:, start_sample:end_sample]

            self.total_windows += 1
            max_amp = float(np.abs(window).max())
            if max_amp > self.max_amplitude_uv:
                self.rejected_windows_amp += 1
                continue

            self.valid_windows += 1

            segments.append(
                {
                    "data": window,
                    "label": label,
                    "start_time": float(t_start),
                    "end_time": float(t_end),
                    "session_id": session_id,
                    "session_name": session_name,
                    "run_name": run_name,
                    "trial_id": trial_counter,
                }
            )
            trial_counter += 1

        return segments, ch_names

    # ------------------------------------------------------------------
    # Build functions
    # ------------------------------------------------------------------

    def _report_stats(self, subject_id: str) -> None:
        """Print basic window statistics for a subject."""
        print(f"Subject {subject_id} window statistics:")
        print(f"  Total windows: {self.total_windows}")
        print(f"  Valid windows: {self.valid_windows}")
        print(f"  Rejected (amplitude): {self.rejected_windows_amp}")
        print(f"  Rejected (ambiguous): {self.rejected_windows_ambiguous}")

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: BIDS subject identifier (e.g., 'sub-04').

        Returns:
            Path to output HDF5 file (as string).
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building ChineseEEG reading dataset")

        # Reset counters
        self.total_windows = 0
        self.valid_windows = 0
        self.rejected_windows_amp = 0
        self.rejected_windows_ambiguous = 0

        runs = self._find_runs_for_subject(subject_id)
        if not runs:
            raise FileNotFoundError(f"No EEG runs found for subject {subject_id}")

        all_segments: List[Dict] = []
        ch_names: Optional[List[str]] = None

        for run_info in runs:
            vhdr = run_info["vhdr"]
            events_tsv: Optional[Path] = run_info["events"]  # type: ignore[assignment]
            session_name = run_info["session"]
            run_name = run_info["run_name"]
            # Map session string to an integer (order is arbitrary but consistent)
            try:
                session_id = int(
                    "".join(ch for ch in session_name if ch.isdigit())
                ) if any(ch.isdigit() for ch in session_name) else 0
            except Exception:
                session_id = 0

            print(f"Processing {vhdr.name} (session={session_name})")

            if events_tsv is None or not events_tsv.exists():
                print(f"  Warning: events.tsv not found for run {vhdr.name}, skipping")
                continue

            try:
                events_df = self._load_events_tsv(events_tsv)
                reading_intervals = self._build_reading_intervals_from_events(events_df)
            except Exception as e:
                print(f"  Warning: failed to parse events for {vhdr.name}: {e}")
                continue

            raw = self._read_raw(vhdr)
            if raw is None:
                continue

            segments, run_ch_names = self._segment_run(
                raw=raw,
                reading_intervals=reading_intervals,
                session_name=session_name,
                run_name=run_name,
                session_id=session_id,
            )

            if not segments:
                print(f"  Warning: no valid segments for run {vhdr.name}")
                continue

            all_segments.extend(segments)

            if ch_names is None:
                ch_names = run_ch_names
                if self._dataset_channels is None:
                    self._dataset_channels = ch_names

        if not all_segments or ch_names is None:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=CHINESEEEG_READING_INFO.dataset_name,
            task_type=CHINESEEEG_READING_INFO.task_type.value,
            downstream_task_type=CHINESEEEG_READING_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CHINESEEEG_READING_INFO.num_labels,
            category_list=CHINESEEEG_READING_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=CHINESEEEG_READING_INFO.montage,
        )

        # Prepare output path
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{subject_id}.h5"

        # Write HDF5 (one segment per trial, similar to LongitudinalEEGBuilder)
        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for seg_idx, seg in enumerate(all_segments):
                trial_attrs = TrialAttrs(
                    trial_id=seg["trial_id"],
                    session_id=seg["session_id"],
                    task_name=seg["session_name"],
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    time_length=self.window_sec,
                    label=np.array([seg["label"]], dtype=int),
                )
                writer.add_segment(trial_name, segment_attrs, seg["data"])

        self._report_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: Dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        channels = self._dataset_channels if self._dataset_channels else CHINESEEEG_READING_INFO.channels

        info = {
            "dataset": {
                "name": CHINESEEEG_READING_INFO.dataset_name,
                "description": "ChineseEEG (ds004952) reading vs non-reading binary classification dataset",
                "task_type": str(CHINESEEEG_READING_INFO.task_type.value),
                "downstream_task": str(CHINESEEEG_READING_INFO.downstream_task_type.value),
                "num_labels": CHINESEEEG_READING_INFO.num_labels,
                "category_list": CHINESEEEG_READING_INFO.category_list,
                "original_sampling_rate": None,  # 1000 Hz in raw, resampled to target_sfreq
                "channels": channels,
                "channel_count": len(channels),
                "montage": CHINESEEEG_READING_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds004952",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "reading_min_overlap": self.reading_min_overlap,
                "nonreading_max_overlap": self.nonreading_max_overlap,
                "include_preface_as_reading": self.include_preface_as_reading,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_all(self, subject_ids: Optional[List[str]] = None) -> List[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs (e.g., ['sub-04', ...]); None = all.

        Returns:
            List of output file paths.
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths: List[str] = []
        failed_subjects: List[str] = []
        all_total = 0
        all_valid = 0
        all_reject_amp = 0
        all_reject_amb = 0

        for sid in subject_ids:
            try:
                path = self.build_subject(sid)
                output_paths.append(path)
                all_total += self.total_windows
                all_valid += self.valid_windows
                all_reject_amp += self.rejected_windows_amp
                all_reject_amb += self.rejected_windows_ambiguous
            except Exception as e:  # pragma: no cover - robustness
                print(f"Error processing subject {sid}: {e}")
                failed_subjects.append(sid)
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal windows: {all_total}")
        print(f"Valid windows: {all_valid}")
        print(f"Rejected (amplitude): {all_reject_amp}")
        print(f"Rejected (ambiguous): {all_reject_amb}")
        if all_total > 0:
            print(f"Valid rate: {all_valid / all_total * 100:.1f}%")
        print("=" * 50)

        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_windows": all_total,
            "valid_windows": all_valid,
            "rejected_windows_amp": all_reject_amp,
            "rejected_windows_ambiguous": all_reject_amb,
        }
        self._save_dataset_info(stats)

        return output_paths


def build_chineseeeg_reading(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function to build ChineseEEG reading vs non‑reading dataset.

    Args:
        raw_data_dir: Root of BIDS ds004952 (contains sub-*/).
        output_dir: Output directory for HDF5 files.
        subject_ids: List of subject IDs (e.g., ['sub-04']); None = all.
        **kwargs: Additional arguments passed to ChineseEEGReadingBuilder.
    """
    builder = ChineseEEGReadingBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ChineseEEG (ds004952) reading vs non-reading HDF5 dataset"
    )
    parser.add_argument("raw_data_dir", help="Root directory of ds004952 BIDS dataset")
    parser.add_argument(
        "--output_dir", default="./hdf5", help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., sub-04 sub-05). Default: all subjects.",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=256.0,
        help="Target sampling frequency (Hz)",
    )
    parser.add_argument(
        "--window_sec", type=float, default=2.0, help="Window length in seconds"
    )
    parser.add_argument(
        "--stride_sec", type=float, default=1.0, help="Stride length in seconds"
    )
    parser.add_argument(
        "--filter_low",
        type=float,
        default=0.5,
        help="Low cutoff frequency for bandpass filter (Hz)",
    )
    parser.add_argument(
        "--filter_high",
        type=float,
        default=40.0,
        help="High cutoff frequency for bandpass filter (Hz)",
    )
    parser.add_argument(
        "--filter_notch",
        type=float,
        default=50.0,
        help="Notch filter frequency (Hz, 0 to disable)",
    )
    parser.add_argument(
        "--max_amplitude_uv",
        type=float,
        default=DEFAULT_MAX_AMPLITUDE_UV,
        help="Maximum amplitude threshold in µV",
    )
    parser.add_argument(
        "--reading_min_overlap",
        type=float,
        default=0.7,
        help="Minimum overlap proportion with reading intervals to label as reading",
    )
    parser.add_argument(
        "--nonreading_max_overlap",
        type=float,
        default=0.3,
        help="Maximum overlap proportion with reading intervals to label as non-reading",
    )
    parser.add_argument(
        "--include_preface_as_reading",
        action="store_true",
        help="Treat preface (PRES–PREE) as reading",
    )

    args = parser.parse_args()

    build_chineseeeg_reading(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subject_ids=args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        reading_min_overlap=args.reading_min_overlap,
        nonreading_max_overlap=args.nonreading_max_overlap,
        include_preface_as_reading=args.include_preface_as_reading,
    )

