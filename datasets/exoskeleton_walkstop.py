"""
EEG-Controlled Exoskeleton for Walking and Standing Dataset Builder.

EEG-Controlled Exoskeleton for Walking and Standing
A Longitudinal Motor Imagery Study in Healthy Adults
- 7 subjects
- 9 sessions per subject
- 2 classes: Walk (motor imagery), Stop (motor imagery)
- 100 Hz sampling rate
- 60 EEG channels + 4 EOG channels
- EDF format
- Tasks: walk6min, stop6min (6-minute motor imagery tasks)
- https://openneuro.org/datasets/ds006940

Task: Binary classification (Walk vs Stop motor imagery)

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)
"""

import os
import json
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


# Exoskeleton Walk/Stop Dataset Configuration
EXOSKELETON_INFO = DatasetInfo(
    dataset_name="Exoskeleton_WalkStop",
    task_type=DatasetTaskType.MOTOR_IMAGINARY,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["Stop", "Walk"],  # 0=Stop, 1=Walk
    sampling_rate=100.0,  # Original sampling rate
    montage="10_20",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Task names for walk/stop classification
WALK_STOP_TASKS = ["walk6min", "stop6min"]

# Label mapping: task name -> class index
TASK_LABEL_MAP = {
    "stop6min": 0,  # Stop motor imagery
    "walk6min": 1,  # Walk motor imagery
}

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.
    
    Uses robust statistics (percentile) instead of max to avoid noise/artifact interference.
    
    Args:
        data: Input data array (shape: n_channels x n_samples)
    
    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    mad = np.median(abs_data)
    if mad > 0:
        mad_based_estimate = 3 * mad
        max_amp = max(max_amp, mad_based_estimate)
    
    if max_amp > 1e-2:  # > 0.01, likely microvolts (µV)
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts (mV)
        return data / 1e3, "mV"
    else:  # likely already Volts (V)
        return data, "V"


class ExoskeletonWalkStopBuilder:
    """Builder for Exoskeleton Walk/Stop dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,  # 60 Hz for USA
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        task_filter: List[str] = None,  # Filter specific tasks (None = all tasks)
    ):
        """
        Initialize Exoskeleton Walk/Stop builder.

        Args:
            raw_data_dir: Directory containing raw files (BIDS format)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds for segmentation
            stride_sec: Stride length in seconds for segmentation
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (60 Hz for USA)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            task_filter: List of task names to process (None = all: walk6min, stop6min)
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "Exoskeleton_WalkStop":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "Exoskeleton_WalkStop"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 100.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.task_filter = task_filter if task_filter else WALK_STOP_TASKS

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = None

    def get_subject_ids(self) -> List[int]:
        """Get list of subject IDs (1-7)."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            try:
                sub_id = int(sub_dir.name.split("-")[1])
                # Check if any session has walk/stop task files
                for ses_dir in sub_dir.glob("ses-*"):
                    eeg_dir = ses_dir / "eeg"
                    if eeg_dir.exists():
                        for task_name in self.task_filter:
                            edf_file = eeg_dir / f"{sub_dir.name}_{ses_dir.name}_task-{task_name}_eeg.edf"
                            if edf_file.exists():
                                subject_ids.append(sub_id)
                                break
                        if sub_id in subject_ids:
                            break
                    if sub_id in subject_ids:
                        break
            except (ValueError, IndexError):
                continue
        return sorted(set(subject_ids))

    def _find_files(self, subject_id: int) -> List[Dict[str, Path]]:
        """
        Find all walk/stop task files for a subject across all sessions.
        
        Returns:
            List of dicts with keys: 'edf', 'task', 'session_id', 'session_name'
        """
        sub_dir = self.raw_data_dir / f"sub-{subject_id:02d}"
        files = []
        
        # Find all sessions
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses_name = ses_dir.name  # e.g., "ses-01"
            ses_id = int(ses_name.split("-")[1])  # Extract number from "ses-01" -> 1
            
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            # Find walk/stop task files
            for task_name in self.task_filter:
                edf_file = eeg_dir / f"{sub_dir.name}_{ses_name}_task-{task_name}_eeg.edf"
                
                if edf_file.exists():
                    files.append({
                        'edf': edf_file,
                        'task': task_name,
                        'session_id': ses_id,
                        'session_name': ses_name,
                    })
        
        return files

    def _read_raw(self, edf_file: Path):
        """
        Read EDF file and convert to MNE Raw object.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EDF files")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            
            # Drop non-EEG channels if needed (keep only EEG channels)
            # EOG channels are typically named with "EOG" or "EOG*"
            eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False)
            if len(eeg_picks) > 0:
                raw.pick(eeg_picks)
            
            # Check if data is in reasonable range for Volts
            if hasattr(raw, '_data') and raw._data is not None:
                max_amp = np.abs(raw._data).max()
                # If max amplitude > 10 V, likely in µV
                if max_amp > 10.0:
                    raw._data = raw._data / 1e6
                    print(f"  Detected unit: µV (max={max_amp:.2e}), converted to V")
                elif max_amp > 1.0:
                    # Between 1-10 V, could be mV
                    raw._data = raw._data / 1e3
                    print(f"  Detected unit: mV (max={max_amp:.2e}), converted to V")
                # Otherwise, assume already in Volts
            
            return raw
        except Exception as e:
            print(f"  Warning: Failed to read {edf_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Check if we need to resample before notch filter
        # For low sampling rates (e.g., 100Hz), notch filter at 60Hz cannot be applied
        # So we resample first, then apply filters
        current_sfreq = raw.info['sfreq']
        nyquist = current_sfreq / 2.0
        
        # If notch filter frequency is too close to or exceeds Nyquist, resample first
        if self.filter_notch > 0 and self.filter_notch >= nyquist * 0.9:
            # Resample first to allow notch filter
            if current_sfreq != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)
                nyquist = raw.info['sfreq'] / 2.0
        
        # Notch filter (60 Hz for USA) - only if notch frequency is less than Nyquist
        if self.filter_notch > 0:
            if self.filter_notch < nyquist:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            else:
                print(f"  Warning: Skipping notch filter ({self.filter_notch}Hz) - exceeds Nyquist ({nyquist:.1f}Hz)")

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if not done earlier
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """
        Validate segment amplitude.

        Args:
            segment_data: Segment data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (1-7)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building Exoskeleton Walk/Stop dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No walk/stop task files found for subject {subject_id}")
        
        all_segments = []
        ch_names = None
        trial_counter = 0

        # Process each task file
        for file_info in files:
            edf_file = file_info['edf']
            task_name = file_info['task']
            session_id = file_info['session_id']
            
            print(f"Reading {edf_file.name} (Task: {task_name}, Session: {session_id})")
            raw = self._read_raw(edf_file)
            if raw is None:
                continue
            
            raw = self._preprocess(raw)
            
            if ch_names is None:
                ch_names = raw.ch_names
                # Store channel names for dataset info (first subject sets it)
                if self._dataset_channels is None:
                    self._dataset_channels = ch_names
            
            # Get data (in Volts)
            data = raw.get_data()  # shape: (n_channels, n_samples)
            n_channels, n_samples = data.shape
            
            # Convert to µV
            data_uv = data * 1e6
            
            # Get task label
            task_label = TASK_LABEL_MAP[task_name]
            
            # Segment into windows (sliding window)
            for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end_sample = start_sample + self.window_samples
                
                # Ensure we don't exceed data bounds
                if end_sample > n_samples:
                    break
                
                segment_data_uv = data_uv[:, start_sample:end_sample]
                
                # Validate segment shape
                if segment_data_uv.shape[1] != self.window_samples:
                    continue
                
                # Validate segment amplitude
                self.total_segments += 1
                if not self._validate_segment(segment_data_uv):
                    self.rejected_segments += 1
                    continue
                
                self.valid_segments += 1
                
                # Calculate time
                start_time = start_sample / self.target_sfreq
                end_time = start_time + self.window_sec
                
                all_segments.append({
                    'data': segment_data_uv,  # Store in µV
                    'trial_id': trial_counter,
                    'session_id': session_id,
                    'task': task_name,
                    'label': task_label,
                    'start_time': start_time,
                    'end_time': end_time,
                })
                
                trial_counter += 1

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=EXOSKELETON_INFO.dataset_name,
            task_type=EXOSKELETON_INFO.task_type.value,
            downstream_task_type=EXOSKELETON_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=EXOSKELETON_INFO.num_labels,
            category_list=EXOSKELETON_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
        )

        # Create output file
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for seg_idx, segment in enumerate(all_segments):
                trial_attrs = TrialAttrs(
                    trial_id=segment['trial_id'],
                    session_id=segment['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,  # Single segment per trial
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=np.array([segment['label']]),
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available
        channels = self._dataset_channels if self._dataset_channels else EXOSKELETON_INFO.channels
        
        info = {
            "dataset": {
                "name": EXOSKELETON_INFO.dataset_name,
                "description": "EEG-Controlled Exoskeleton for Walking and Standing - Walk vs Stop Motor Imagery Classification",
                "task_type": str(EXOSKELETON_INFO.task_type.value),
                "downstream_task": str(EXOSKELETON_INFO.downstream_task_type.value),
                "num_labels": EXOSKELETON_INFO.num_labels,
                "category_list": EXOSKELETON_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": EXOSKELETON_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds006940",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "task_filter": self.task_filter,
                "max_amplitude_uv": self.max_amplitude_uv,
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
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_all(self, subject_ids: List[int] = None) -> List[str]:
        """
        Build HDF5 files for all subjects.

        Args:
            subject_ids: List of subject IDs to process (None = all)

        Returns:
            List of output file paths
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                failed_subjects.append(subject_id)
                import traceback
                traceback.print_exc()

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_exoskeleton_walkstop(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: List[int] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function to build Exoskeleton Walk/Stop dataset.

    Args:
        raw_data_dir: Directory containing raw files (BIDS format)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for ExoskeletonWalkStopBuilder

    Returns:
        List of output file paths
    """
    builder = ExoskeletonWalkStopBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Exoskeleton Walk/Stop HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (BIDS format)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--tasks", nargs="+", choices=WALK_STOP_TASKS, help="Tasks to process (default: all)")
    args = parser.parse_args()

    build_exoskeleton_walkstop(args.raw_data_dir, args.output_dir, args.subjects, task_filter=args.tasks)
