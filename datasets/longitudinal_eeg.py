"""
Longitudinal EEG Reliability Dataset Builder.

Longitudinal EEG Test-Retest Reliability in Healthy Individuals
- 43 subjects
- 4 sessions per subject (ses-V0, ses-V1, ses-V2, ses-V3)
- 2 tasks per session: CE (eyes closed) and OE (eyes open)
- Resting-state EEG
- 1000 Hz sampling rate
- BrainVision format (.vhdr, .eeg, .vmrk)
- https://openneuro.org/datasets/ds007176

Task: Cross-session person identification (43 classes)
- Each subject is assigned a unique label from 1-43
- Data from all sessions are used for training
- Classification task: identify the subject from EEG signals

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Data is already in Volts according to channels.tsv
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


# Longitudinal EEG Reliability Dataset Configuration
LONGITUDINAL_EEG_INFO = DatasetInfo(
    dataset_name="Longitudinal_EEG_Reliability",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=43,  # 43 subjects
    category_list=[],  # Will be populated from subject IDs
    sampling_rate=200.0,  # Target sampling rate (downsampled from 1000 Hz)
    montage="10_20",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Task names
TASK_NAMES = ["CE", "OE"]  # CE: eyes closed, OE: eyes open

# Session names
SESSION_NAMES = ["ses-V0", "ses-V1", "ses-V2", "ses-V3"]


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
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


class LongitudinalEEGBuilder:
    """Builder for Longitudinal EEG Reliability dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,  # 60 Hz for Colombia
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        task_filter: List[str] = None,  # Filter specific tasks (None = all tasks)
    ):
        """
        Initialize Longitudinal EEG Reliability builder.

        Args:
            raw_data_dir: Directory containing raw files (BIDS format)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (60 Hz for Colombia)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            task_filter: List of task names to process (None = all tasks: CE, OE)
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "Longitudinal_EEG_Reliability":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "Longitudinal_EEG_Reliability"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.task_filter = task_filter if task_filter else TASK_NAMES

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = None

        # Subject ID to label mapping (1-43)
        self._subject_to_label: Dict[str, int] = {}
        self._label_to_subject: Dict[int, str] = {}
        self._init_subject_mapping()

    def _init_subject_mapping(self):
        """Initialize subject ID to label mapping (1-43)."""
        participants_file = self.raw_data_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found: {participants_file}")
        
        participants_df = pd.read_csv(participants_file, sep='\t')
        subject_ids = sorted(participants_df['participant_id'].tolist())
        
        # Map each subject to a unique label from 1 to 43
        category_list = []
        for idx, subject_id in enumerate(subject_ids, start=1):
            self._subject_to_label[subject_id] = idx
            self._label_to_subject[idx] = subject_id
            # Store subject ID in category_list (0-indexed, so idx-1)
            category_list.append(subject_id)
        
        # Update category_list with actual subject IDs
        LONGITUDINAL_EEG_INFO.category_list = category_list
        
        print(f"Initialized subject mapping: {len(self._subject_to_label)} subjects")

    def get_subject_ids(self) -> List[str]:
        """Get list of subject IDs from participants.tsv."""
        participants_file = self.raw_data_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found: {participants_file}")
        
        participants_df = pd.read_csv(participants_file, sep='\t')
        return sorted(participants_df['participant_id'].tolist())

    def _find_files(self, subject_id: str) -> List[Dict[str, Path]]:
        """
        Find all task files for a subject across all sessions.
        
        Returns:
            List of dicts with keys: 'vhdr', 'task', 'session_id', 'session_name'
        """
        sub_dir = self.raw_data_dir / subject_id
        files = []
        
        # Find all sessions
        for session_name in SESSION_NAMES:
            ses_dir = sub_dir / session_name
            if not ses_dir.exists():
                continue
            
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            # Find all task files
            for task_name in self.task_filter:
                vhdr_file = eeg_dir / f"{subject_id}_{session_name}_task-{task_name}_eeg.vhdr"
                
                if vhdr_file.exists():
                    # Extract session number from session_name (e.g., "ses-V0" -> 0)
                    session_id = int(session_name.split("-V")[1])
                    files.append({
                        'vhdr': vhdr_file,
                        'task': task_name,
                        'session_id': session_id,
                        'session_name': session_name,
                    })
        
        return files

    def _read_raw(self, vhdr_file: Path):
        """
        Read BrainVision file and convert to MNE Raw object.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading BrainVision files")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=True, verbose=False)
            
            # Check if data is in reasonable range for Volts
            # According to channels.tsv, data should be in Volts already
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
            print(f"  Warning: Failed to read {vhdr_file.name}: {e}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Keep only EEG channels (drop EOG/ECG/HEO/VEO and other non-EEG types)
        # This ensures dataset_info.json only reports pure EEG channels.
        try:
            raw.pick_types(eeg=True, exclude=[])
        except Exception:
            # Fallback: if pick_types fails for any reason, continue with all channels
            pass

        # Notch filter (60 Hz for Colombia)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
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

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (e.g., "sub-G2001")

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building Longitudinal EEG Reliability dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Get subject label (1-43)
        if subject_id not in self._subject_to_label:
            raise ValueError(f"Subject {subject_id} not found in participants.tsv")
        subject_label = self._subject_to_label[subject_id]

        # Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No task files found for subject {subject_id}")
        
        all_segments = []
        ch_names = None
        trial_counter = 0

        # Process each task file
        for file_info in files:
            vhdr_file = file_info['vhdr']
            task_name = file_info['task']
            session_id = file_info['session_id']
            
            print(f"Reading {vhdr_file.name} (Task: {task_name}, Session: {session_id})")
            raw = self._read_raw(vhdr_file)
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
                    'label': subject_label - 1,  # Convert to 0-indexed (0-42)
                    'start_time': start_time,
                    'end_time': end_time,
                })
                
                trial_counter += 1

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=LONGITUDINAL_EEG_INFO.dataset_name,
            task_type=LONGITUDINAL_EEG_INFO.task_type.value,
            downstream_task_type=LONGITUDINAL_EEG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=LONGITUDINAL_EEG_INFO.num_labels,
            category_list=LONGITUDINAL_EEG_INFO.category_list,
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
        # Use subject_id directly (already in format like "sub-G2001")
        output_path = self.output_dir / f"{subject_id}.h5"

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
        channels = self._dataset_channels if self._dataset_channels else LONGITUDINAL_EEG_INFO.channels
        
        info = {
            "dataset": {
                "name": LONGITUDINAL_EEG_INFO.dataset_name,
                "description": "Longitudinal EEG Test-Retest Reliability in Healthy Individuals - Cross-session person identification",
                "task_type": str(LONGITUDINAL_EEG_INFO.task_type.value),
                "downstream_task": str(LONGITUDINAL_EEG_INFO.downstream_task_type.value),
                "num_labels": LONGITUDINAL_EEG_INFO.num_labels,
                "category_list": LONGITUDINAL_EEG_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": LONGITUDINAL_EEG_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds007176",
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

    def build_all(self, subject_ids: List[str] = None) -> List[str]:
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


def build_longitudinal_eeg(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: List[str] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function to build Longitudinal EEG Reliability dataset.

    Args:
        raw_data_dir: Directory containing raw files (BIDS format)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for LongitudinalEEGBuilder

    Returns:
        List of output file paths
    """
    builder = LongitudinalEEGBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Longitudinal EEG Reliability HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (BIDS format)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to process (default: all)")
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, help="Tasks to process (default: all)")
    args = parser.parse_args()

    build_longitudinal_eeg(args.raw_data_dir, args.output_dir, args.subjects, task_filter=args.tasks)
