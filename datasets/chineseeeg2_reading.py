"""
ChineseEEG2 Reading vs Non-Reading Dataset Builder.

This module processes ChineseEEG2 dataset (preprocessed HDF5 format) to produce
a binary classification dataset for reading vs non-reading, following the
benchmark_dataloader standard format.

Data Structure:
- ALIGN_ROOT: Contains label files (windowed_audio_labels.txt) organized by task/run/subject
- EEG_ROOT: Contains preprocessed EEG HDF5 files organized by subject/session/eeg
- Output: Standard benchmark_dataloader HDF5 format with SubjectAttrs/TrialAttrs/SegmentAttrs

The implementation follows the benchmark_dataloader style using:
- SubjectAttrs / TrialAttrs / SegmentAttrs
- HDF5Writer for consistent output format
- One HDF5 file per subject per session
"""

from __future__ import annotations

import glob
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
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

CHINESEEEG2_READING_INFO = DatasetInfo(
    dataset_name="ChineseEEG2_reading_binary",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["non_reading", "reading"],
    sampling_rate=200.0,  # Target sampling rate
    montage="10_20",
    channels=[],  # Will be populated from data
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Label values
LABEL_READING = 1
LABEL_NON_READING = 0


class ChineseEEG2ReadingBuilder:
    """Builder for ChineseEEG2 reading vs non-reading dataset."""

    def __init__(
        self,
        align_root: str,
        eeg_root: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        samples_per_class: Optional[int] = None,  # If None, use all samples; otherwise limit per class
        tasks: Optional[List[str]] = None,  # If None, process all tasks
    ):
        """
        Initialize ChineseEEG2 reading vs non-reading builder.

        Args:
            align_root: Root directory containing label files (windowed_audio_labels.txt)
                       Structure: align_root/task/run-*/subject/windowed_audio_labels.txt
            eeg_root: Root directory containing preprocessed EEG HDF5 files
                     Structure: eeg_root/subject/ses-task/eeg/*.h5
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency (Hz)
            window_sec: Window length in seconds
            filter_low: Low cutoff frequency for bandpass filter (Hz)
            filter_high: High cutoff frequency for bandpass filter (Hz)
            filter_notch: Notch filter frequency (Hz), 0 to disable
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            samples_per_class: Maximum samples per class per subject (None = use all)
            tasks: List of tasks to process (e.g., ['garnettdream', 'littleprince']), None = all
        """
        self.align_root = Path(align_root)
        self.eeg_root = Path(eeg_root)
        output_path = Path(output_dir)
        if output_path.name == CHINESEEEG2_READING_INFO.dataset_name:
            self.output_dir = output_path
        else:
            self.output_dir = output_path / CHINESEEEG2_READING_INFO.dataset_name

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.samples_per_class = samples_per_class
        self.tasks = tasks if tasks else ["garnettdream", "littleprince"]

        self.window_samples = int(self.window_sec * self.target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments_amp = 0

        # Dataset-level channels (first subject sets this)
        self._dataset_channels: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # File finding and loading
    # ------------------------------------------------------------------

    def _find_subjects(self) -> List[str]:
        """Find all subjects from align_root or eeg_root."""
        subjects = set()

        # Try to find from align_root
        if self.align_root.exists():
            for task_dir in self.align_root.glob("*"):
                if not task_dir.is_dir():
                    continue
                for run_dir in task_dir.glob("run-*"):
                    for sub_dir in run_dir.glob("sub-*"):
                        if sub_dir.is_dir():
                            subjects.add(sub_dir.name)

        # Also check eeg_root
        if self.eeg_root.exists():
            for sub_dir in self.eeg_root.glob("sub-*"):
                if sub_dir.is_dir():
                    subjects.add(sub_dir.name)

        return sorted(list(subjects))

    def _find_runs_for_subject_task(
        self, subject_id: str, task: str
    ) -> List[Dict[str, Path]]:
        """
        Find all runs for a subject and task.

        Returns list of dicts with keys:
          - 'label_file': Path to windowed_audio_labels.txt
          - 'eeg_file': Path to EEG HDF5 file
          - 'run_name': Name of the run (e.g., 'run-01')
        """
        runs: List[Dict[str, Path]] = []
        pure_id = subject_id.replace("sub-", "")

        # Find label files in align_root
        task_align_dir = self.align_root / task
        if not task_align_dir.exists():
            return runs

        for run_dir in sorted(task_align_dir.glob("run-*")):
            # Try both pure_id and subject_id as folder names
            label_path = run_dir / pure_id / "windowed_audio_labels.txt"
            if not label_path.exists():
                label_path = run_dir / subject_id / "windowed_audio_labels.txt"
            if not label_path.exists():
                continue

            # Find corresponding EEG file
            run_id_digits = run_dir.name.split("-")[-1]
            eeg_folder = self.eeg_root / subject_id / f"ses-{task}" / "eeg"
            if not eeg_folder.exists():
                eeg_folder = self.eeg_root / pure_id / f"ses-{task}" / "eeg"
            if not eeg_folder.exists():
                continue

            # Try to find matching HDF5 file
            candidates = [f"run-{run_id_digits}_", f"run-{int(run_id_digits):02d}_"]
            eeg_file = None
            for f_name in eeg_folder.glob("*.h5"):
                for cand in candidates:
                    if cand in f_name.name:
                        eeg_file = f_name
                        break
                if eeg_file:
                    break

            if eeg_file and eeg_file.exists():
                runs.append(
                    {
                        "label_file": label_path,
                        "eeg_file": eeg_file,
                        "run_name": run_dir.name,
                    }
                )

        return runs

    def _load_eeg_hdf5(self, h5_path: Path) -> Tuple[Optional[mne.io.Raw], List[str]]:
        """
        Load preprocessed EEG from HDF5 file and return MNE Raw object.

        Returns:
            (raw, ch_names) or (None, []) on error
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for processing EEG data")

        try:
            with h5py.File(h5_path, "r") as f:
                if "eeg_data" not in f:
                    return None, []

                data = f["eeg_data"][:]  # Shape: (n_channels, n_samples)

                # Get sampling rate
                sfreq = self.target_sfreq  # Default
                if "info" in f:
                    if "sfreq" in f["info"]:
                        val = f["info"]["sfreq"][()]
                        sfreq = float(val.item() if hasattr(val, "item") else val)
                    elif "sfreq" in f["info"].attrs:
                        sfreq = float(f["info"].attrs["sfreq"])

                # Get channel names
                ch_names = [f"Ch{i}" for i in range(data.shape[0])]
                if "info" in f and "ch_names" in f["info"]:
                    tmp = f["info"]["ch_names"][:]
                    ch_names = [
                        n.decode("utf-8") if isinstance(n, bytes) else str(n)
                        for n in tmp
                    ]

            # Create MNE Raw object
            # Data is assumed to be in Volts (or already in µV, we'll check)
            # Check if data is already in µV (typical range: 10-1000)
            max_amp = np.abs(data).max()
            if max_amp > 10.0:  # Likely already in µV
                data_volts = data / 1e6
            else:
                data_volts = data  # Assume Volts

            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data_volts, info, verbose=False)

            # Apply filtering if needed
            if self.filter_notch > 0:
                try:
                    raw.notch_filter(freqs=self.filter_notch, verbose=False)
                except Exception:
                    pass

            raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

            # Resample if needed
            if abs(raw.info["sfreq"] - self.target_sfreq) > 0.1:
                raw.resample(self.target_sfreq, verbose=False)

            # Keep only EEG channels (drop EOG/ECG if any)
            try:
                raw.pick_types(eeg=True, exclude=[])
            except Exception:
                pass

            return raw, raw.ch_names

        except Exception as e:
            print(f"  Warning: Failed to load {h5_path.name}: {e}")
            return None, []

    def _parse_label_file(self, label_path: Path) -> List[Tuple[float, int]]:
        """
        Parse windowed_audio_labels.txt file.

        Returns:
            List of (time_sec, label) tuples where label is 0 or 1
        """
        try:
            df = pd.read_csv(label_path, sep="\t")
            df.columns = [c.strip() for c in df.columns]

            if "EEG_time" not in df.columns or "window_label" not in df.columns:
                return []

            df = df.dropna(subset=["window_label"])
            labels = []
            for _, row in df.iterrows():
                time_sec = float(row["EEG_time"])
                label = int(row["window_label"])
                if label in [LABEL_READING, LABEL_NON_READING]:
                    labels.append((time_sec, label))

            return labels

        except Exception as e:
            print(f"  Warning: Failed to parse {label_path.name}: {e}")
            return []

    def _extract_segments(
        self, raw: mne.io.Raw, label_points: List[Tuple[float, int]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract segments from raw data based on label points.

        Returns:
            (reading_segments, non_reading_segments) - each is a list of arrays (n_channels, window_samples)
        """
        data_uv = raw.get_data() * 1e6  # Convert to µV
        n_samples = data_uv.shape[1]
        sfreq = raw.info["sfreq"]

        reading_segs: List[np.ndarray] = []
        non_reading_segs: List[np.ndarray] = []

        for time_sec, label in label_points:
            start_idx = int(time_sec * sfreq)
            end_idx = start_idx + self.window_samples

            if start_idx < 0 or end_idx > n_samples:
                continue

            seg = data_uv[:, start_idx:end_idx]

            # Validate amplitude
            if np.max(np.abs(seg)) > self.max_amplitude_uv:
                self.rejected_segments_amp += 1
                continue

            self.total_segments += 1
            self.valid_segments += 1

            if label == LABEL_READING:
                reading_segs.append(seg)
            elif label == LABEL_NON_READING:
                non_reading_segs.append(seg)

        return reading_segs, non_reading_segs

    # ------------------------------------------------------------------
    # Build functions
    # ------------------------------------------------------------------

    def build_subject_task(
        self, subject_id: str, task: str
    ) -> Optional[str]:
        """
        Build HDF5 file for a single subject and task.

        Args:
            subject_id: Subject identifier (e.g., 'sub-f1')
            task: Task name (e.g., 'garnettdream')

        Returns:
            Path to output HDF5 file, or None if failed
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building ChineseEEG2 dataset")

        # Reset counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments_amp = 0

        runs = self._find_runs_for_subject_task(subject_id, task)
        if not runs:
            print(f"  Warning: No runs found for {subject_id} task {task}")
            return None

        all_reading_segs: List[np.ndarray] = []
        all_non_reading_segs: List[np.ndarray] = []
        ch_names: Optional[List[str]] = None

        # Process each run
        for run_info in runs:
            label_path = run_info["label_file"]
            eeg_path = run_info["eeg_file"]
            run_name = run_info["run_name"]

            print(f"  Processing {run_name}...")

            # Load EEG
            raw, run_ch_names = self._load_eeg_hdf5(eeg_path)
            if raw is None:
                continue

            if ch_names is None:
                ch_names = run_ch_names
                if self._dataset_channels is None:
                    self._dataset_channels = ch_names

            # Load labels
            label_points = self._parse_label_file(label_path)
            if not label_points:
                continue

            # Extract segments
            reading_segs, non_reading_segs = self._extract_segments(raw, label_points)
            all_reading_segs.extend(reading_segs)
            all_non_reading_segs.extend(non_reading_segs)

        if not all_reading_segs and not all_non_reading_segs:
            print(f"  Warning: No valid segments for {subject_id} task {task}")
            return None

        if ch_names is None:
            print(f"  Warning: No channel names found for {subject_id} task {task}")
            return None

        # Balance samples if requested
        random.shuffle(all_reading_segs)
        random.shuffle(all_non_reading_segs)

        if self.samples_per_class is not None:
            target_count = min(
                len(all_reading_segs),
                len(all_non_reading_segs),
                self.samples_per_class,
            )
            all_reading_segs = all_reading_segs[:target_count]
            all_non_reading_segs = all_non_reading_segs[:target_count]

        # Combine and shuffle
        all_segments: List[Dict] = []
        for seg in all_reading_segs:
            all_segments.append({"data": seg, "label": LABEL_READING})
        for seg in all_non_reading_segs:
            all_segments.append({"data": seg, "label": LABEL_NON_READING})

        random.shuffle(all_segments)

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=f"{CHINESEEEG2_READING_INFO.dataset_name}_{task}",
            task_type=CHINESEEEG2_READING_INFO.task_type.value,
            downstream_task_type=CHINESEEEG2_READING_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CHINESEEEG2_READING_INFO.num_labels,
            category_list=CHINESEEEG2_READING_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=CHINESEEEG2_READING_INFO.montage,
        )

        # Prepare output path
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for task
        task_output_dir = self.output_dir / f"ses-{task}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = task_output_dir / f"{subject_id}.h5"

        # Write HDF5 using HDF5Writer
        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial_idx, seg_dict in enumerate(all_segments):
                trial_attrs = TrialAttrs(
                    trial_id=trial_idx,
                    session_id=0,  # Single session per task
                    task_name=task,
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,  # One segment per trial
                    start_time=0.0,  # Time info not available from label file
                    end_time=self.window_sec,
                    time_length=self.window_sec,
                    label=np.array([seg_dict["label"]], dtype=int),
                )
                writer.add_segment(trial_name, segment_attrs, seg_dict["data"])

        print(
            f"  ✅ {subject_id} ({task}): {len(all_segments)} segments "
            f"(reading={len(all_reading_segs)}, non_reading={len(all_non_reading_segs)})"
        )

        return str(output_path)

    def _save_dataset_info(self, stats: Dict, task: str) -> None:
        """Save dataset info and processing parameters to JSON."""
        channels = (
            self._dataset_channels
            if self._dataset_channels
            else CHINESEEEG2_READING_INFO.channels
        )

        info = {
            "dataset": {
                "name": f"{CHINESEEEG2_READING_INFO.dataset_name}_{task}",
                "description": f"ChineseEEG2 reading vs non-reading binary classification dataset (task: {task})",
                "task_type": str(CHINESEEEG2_READING_INFO.task_type.value),
                "downstream_task": str(CHINESEEEG2_READING_INFO.downstream_task_type.value),
                "num_labels": CHINESEEEG2_READING_INFO.num_labels,
                "category_list": CHINESEEEG2_READING_INFO.category_list,
                "original_sampling_rate": None,  # Varies by source file
                "channels": channels,
                "channel_count": len(channels),
                "montage": CHINESEEEG2_READING_INFO.montage,
                "source_url": "https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "samples_per_class": self.samples_per_class,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        task_output_dir = self.output_dir / f"ses-{task}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        json_path = task_output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_all(
        self, subject_ids: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Build HDF5 files for all subjects and tasks.

        Args:
            subject_ids: List of subject IDs to process (None = all)

        Returns:
            Dict mapping task names to lists of output file paths
        """
        if subject_ids is None:
            subject_ids = self._find_subjects()

        output_paths_by_task: Dict[str, List[str]] = {task: [] for task in self.tasks}
        failed_subjects: List[str] = []

        all_total = 0
        all_valid = 0
        all_reject_amp = 0

        for task in self.tasks:
            print(f"\n{'='*60}")
            print(f"Processing task: {task}")
            print(f"{'='*60}")

            task_total = 0
            task_valid = 0
            task_reject_amp = 0
            task_success = 0
            task_failed = 0

            for subject_id in subject_ids:
                try:
                    path = self.build_subject_task(subject_id, task)
                    if path:
                        output_paths_by_task[task].append(path)
                        task_success += 1
                        task_total += self.total_segments
                        task_valid += self.valid_segments
                        task_reject_amp += self.rejected_segments_amp
                    else:
                        task_failed += 1
                        failed_subjects.append(f"{subject_id}_{task}")
                except Exception as e:
                    print(f"Error processing {subject_id} ({task}): {e}")
                    task_failed += 1
                    failed_subjects.append(f"{subject_id}_{task}")
                    import traceback

                    traceback.print_exc()

            all_total += task_total
            all_valid += task_valid
            all_reject_amp += task_reject_amp

            # Save task-specific dataset info
            stats = {
                "total_subjects": len(subject_ids),
                "successful_subjects": task_success,
                "failed_subjects": task_failed,
                "total_segments": task_total,
                "valid_segments": task_valid,
                "rejected_segments_amp": task_reject_amp,
            }
            self._save_dataset_info(stats, task)

            print(f"\nTask {task} summary:")
            print(f"  Successful: {task_success}/{len(subject_ids)}")
            print(f"  Failed: {task_failed}/{len(subject_ids)}")
            print(f"  Valid segments: {task_valid}")
            print(f"  Rejected (amplitude): {task_reject_amp}")

        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total subjects processed: {len(subject_ids)}")
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Total segments: {all_total}")
        print(f"Valid segments: {all_valid}")
        print(f"Rejected (amplitude): {all_reject_amp}")
        if all_total > 0:
            print(f"Valid rate: {all_valid / all_total * 100:.1f}%")
        print("=" * 60)

        return output_paths_by_task


def build_chineseeeg2_reading(
    align_root: str,
    eeg_root: str,
    output_dir: str = "./hdf5",
    subject_ids: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, List[str]]:
    """
    Convenience function to build ChineseEEG2 reading vs non-reading dataset.

    Args:
        align_root: Root directory containing label files
        eeg_root: Root directory containing preprocessed EEG HDF5 files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments passed to ChineseEEG2ReadingBuilder

    Returns:
        Dict mapping task names to lists of output file paths
    """
    builder = ChineseEEG2ReadingBuilder(align_root, eeg_root, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ChineseEEG2 reading vs non-reading HDF5 dataset"
    )
    parser.add_argument(
        "align_root",
        help="Root directory containing label files (windowed_audio_labels.txt)",
    )
    parser.add_argument(
        "eeg_root",
        help="Root directory containing preprocessed EEG HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        default="./hdf5",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., sub-f1 sub-f2). Default: all subjects.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["garnettdream", "littleprince"],
        help="Tasks to process. Default: all tasks.",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=200.0,
        help="Target sampling frequency (Hz)",
    )
    parser.add_argument(
        "--window_sec", type=float, default=1.0, help="Window length in seconds"
    )
    parser.add_argument(
        "--filter_low",
        type=float,
        default=0.1,
        help="Low cutoff frequency for bandpass filter (Hz)",
    )
    parser.add_argument(
        "--filter_high",
        type=float,
        default=75.0,
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
        "--samples_per_class",
        type=int,
        default=None,
        help="Maximum samples per class per subject (None = use all)",
    )

    args = parser.parse_args()

    build_chineseeeg2_reading(
        align_root=args.align_root,
        eeg_root=args.eeg_root,
        output_dir=args.output_dir,
        subject_ids=args.subjects,
        tasks=args.tasks,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        samples_per_class=args.samples_per_class,
    )
