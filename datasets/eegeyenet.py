"""
EEGEyeNet Dataset Builder.

EEGEyeNet Dataset - Eye Movement Prediction from EEG
- BIDS format dataset
- 500 Hz sampling rate
- 129 channels (E1-E128 + Cz)
- Task: Eye movement prediction (EOG)
- Labels: 27 classes (eye movement directions/positions)
- https://openneuro.org/datasets/ds005872/versions/1.0.0
- Reference: Kastrati, A., Płomecka, M. B., Pascual, D., Wolf, L., Gillioz, V., 
             Wattenhofer, R., & Langer, N. (2021). EEGEyeNet: A Simultaneous 
             Electroencephalography and Eye-tracking Dataset and Benchmark for 
             Eye Movement Prediction (Version 2). arXiv.
"""

import os
import re
import json
import warnings
from pathlib import Path
from datetime import datetime
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
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

# Dataset Configuration
EEGEYENET_INFO = DatasetInfo(
    dataset_name="EEGEyeNet",
    task_type=DatasetTaskType.EOG,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=27,  # 27 eye movement directions/positions
    category_list=[str(i) for i in range(1, 28)],  # Labels 1-27
    sampling_rate=500.0,
    montage="10_10",  # High-density EEG
    channels=[f"E{i}" for i in range(1, 129)] + ["Cz"],  # E1-E128 + Cz
)

# Channels to remove (if any)
REMOVE_CHANNELS = []

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE.

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    max_amp = np.abs(data).max()

    if max_amp > 1e-2:  # > 0.01, likely microvolts
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts
        return data / 1e3, "mV"
    else:  # likely already Volts
        return data, "V"


class EEGEyeNetBuilder:
    """Builder for EEGEyeNet dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # Europe/Switzerland uses 50Hz
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        task_type: str = "classification",  # "classification" or "regression"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "EEGEyeNet"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 500.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.task_type = task_type

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs from BIDS directory structure."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            # Extract subject ID from directory name (e.g., "sub-EP10" -> 10)
            match = re.search(r'sub-EP?(\d+)', sub_dir.name)
            if match:
                subject_ids.append(int(match.group(1)))
        return sorted(subject_ids)

    def _find_files(self, subject_id: int) -> list[dict]:
        """
        Find all EDF files for a subject in BIDS format.
        
        Returns:
            List of dicts with keys: 'edf_path', 'events_path', 'session_id', 'run_id'
        """
        # Try different subject ID formats
        subject_patterns = [
            f"sub-EP{subject_id:02d}",
            f"sub-{subject_id:02d}",
            f"sub-EP{subject_id}",
            f"sub-{subject_id}",
        ]
        
        files = []
        for pattern in subject_patterns:
            subject_dir = self.raw_data_dir / pattern
            if not subject_dir.exists():
                continue
            
            # Find all sessions
            for session_dir in sorted(subject_dir.glob("ses-*")):
                session_id = int(re.search(r'ses-(\d+)', session_dir.name).group(1))
                eeg_dir = session_dir / "eeg"
                
                if not eeg_dir.exists():
                    continue
                
                # Find all EDF files
                for edf_file in sorted(eeg_dir.glob(f"{pattern}_ses-*_task-*_run-*_eeg.edf")):
                    # Extract run ID
                    run_match = re.search(r'run-(\d+)', edf_file.name)
                    run_id = int(run_match.group(1)) if run_match else 1
                    
                    # Find corresponding events file
                    events_file = edf_file.parent / edf_file.name.replace("_eeg.edf", "_events.tsv")
                    
                    files.append({
                        'edf_path': edf_file,
                        'events_path': events_file if events_file.exists() else None,
                        'session_id': session_id,
                        'run_id': run_id,
                    })
        
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")
        
        return files

    def _read_edf(self, file_path: Path):
        """Read EDF file and convert to MNE Raw object."""
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
        
        # Print channel info
        print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names[:5]}...{raw.ch_names[-5:]}")
        
        # Check if data is already in Volts (BIDS standard)
        # EDF files from BIDS are typically in Volts
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:  # likely already Volts
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")
        
        return raw

    def _load_events(self, events_path: Path) -> pd.DataFrame:
        """Load events from TSV file."""
        if events_path is None or not events_path.exists():
            return None
        
        events_df = pd.read_csv(events_path, sep='\t')
        return events_df

    def _extract_trials_from_events(self, raw, events_df: pd.DataFrame) -> list[dict]:
        """
        Extract trials from events dataframe.
        
        For EEGEyeNet, we extract segments based on trial_type events (1-27).
        Each trial starts at the event onset and ends at the next 'end_cue' or next trial.
        """
        trials = []
        
        if events_df is None:
            # If no events file, treat entire recording as one trial
            trials.append({
                'onset': 0.0,
                'duration': raw.times[-1],
                'label': 0,  # Default label
            })
            return trials
        
        # Filter to get only numeric trial_type events (exclude 'end_cue')
        trial_events = events_df[
            events_df['trial_type'].astype(str).str.isdigit()
        ].copy()
        
        if len(trial_events) == 0:
            # Fallback: use value column if trial_type doesn't have digits
            trial_events = events_df[
                (events_df['trial_type'] != 'end_cue') & 
                (events_df['value'].notna())
            ].copy()
        
        # Sort by onset
        trial_events = trial_events.sort_values('onset').reset_index(drop=True)
        
        for idx, row in trial_events.iterrows():
            onset = row['onset']
            trial_type = str(row['trial_type'])
            
            # Skip 'end_cue' events
            if trial_type == 'end_cue':
                continue
            
            # Try to extract label from trial_type or value
            label = 0  # Default
            try:
                if trial_type.isdigit():
                    label = int(trial_type) - 1  # Convert to 0-indexed (1-27 -> 0-26)
                elif 'value' in row and pd.notna(row['value']):
                    # Use value column if available
                    label = int(row['value']) - 1
            except (ValueError, KeyError, TypeError):
                label = 0
            
            # Ensure label is in valid range (0-26 for 27 classes)
            label = max(0, min(26, label))
            
            # Calculate duration: until next 'end_cue' or next trial, or end of recording
            duration = self.window_sec  # Default duration
            
            # Look for next 'end_cue' event
            next_end_cue = events_df[
                (events_df['trial_type'] == 'end_cue') & 
                (events_df['onset'] > onset)
            ]
            
            if len(next_end_cue) > 0:
                duration = next_end_cue.iloc[0]['onset'] - onset
            elif idx + 1 < len(trial_events):
                # Use next trial onset
                next_onset = trial_events.iloc[idx + 1]['onset']
                duration = next_onset - onset
            else:
                # Use end of recording
                duration = raw.times[-1] - onset
            
            # Ensure duration is positive and reasonable
            if duration <= 0:
                duration = self.window_sec  # Default to window length
            
            # Limit duration to reasonable maximum (e.g., 10 seconds)
            duration = min(duration, 10.0)
            
            trials.append({
                'onset': onset,
                'duration': duration,
                'label': label,
            })
        
        return trials

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop channels if specified
        channels_to_drop = [ch for ch in REMOVE_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
        
        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        
        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        
        return raw

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building EEGEyeNet dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find all files for this subject
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = None
        trial_counter = 0

        # Process each file
        for file_info in files:
            edf_path = file_info['edf_path']
            events_path = file_info['events_path']
            session_id = file_info['session_id']
            run_id = file_info['run_id']

            print(f"Processing {edf_path.name} (session {session_id}, run {run_id})")

            try:
                # Read EDF file
                raw = self._read_edf(edf_path)
                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = raw.ch_names

                # Load events
                events_df = self._load_events(events_path)

                # Extract trials from events
                trials = self._extract_trials_from_events(raw, events_df)
                data = raw.get_data()  # Already in Volts

                # Process each trial
                for trial in trials:
                    onset_sec = trial['onset']
                    duration_sec = trial['duration']
                    label = trial['label']

                    # Convert onset to samples
                    onset_sample = int(onset_sec * self.target_sfreq)
                    duration_samples = int(duration_sec * self.target_sfreq)
                    end_sample = min(onset_sample + duration_samples, data.shape[1])

                    if end_sample <= onset_sample:
                        continue

                    trial_data = data[:, onset_sample:end_sample]

                    all_trials.append({
                        'data': trial_data,  # In Volts
                        'label': label,
                        'trial_id': trial_counter,
                        'session_id': session_id,
                        'run_id': run_id,
                        'onset_time': onset_sec,
                    })
                    trial_counter += 1

            except Exception as e:
                print(f"Error processing {edf_path}: {e}")
                import traceback
                traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=EEGEYENET_INFO.dataset_name,
            task_type=EEGEYENET_INFO.task_type.value,
            downstream_task_type=EEGEYENET_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=EEGEYENET_INFO.num_labels,
            category_list=EEGEYENET_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=EEGEYENET_INFO.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial['data']  # In Volts
                self.total_trials += 1

                # Convert to µV for validation
                trial_data_uv = trial_data * 1e6

                # Validate trial amplitude
                if not self._validate_trial(trial_data_uv):
                    self.rejected_trials += 1
                    print(f"  Skipping trial {trial['trial_id']}: amplitude {np.abs(trial_data_uv).max():.1f} µV > {self.max_amplitude_uv} µV")
                    continue

                self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # Segment into windows
                n_samples = trial_data.shape[1]

                for i_slice, start in enumerate(range(0, n_samples - self.window_samples + 1, self.stride_samples)):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    # Convert from V to µV for export
                    slice_data_uv = slice_data * 1e6

                    # Calculate absolute time
                    seg_start_time = trial['onset_time'] + start / self.target_sfreq
                    seg_end_time = seg_start_time + self.window_sec

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data_uv)

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path} ({self.valid_trials} valid trials)")
        return str(output_path)

    def build_all(self, subject_ids: list[int] = None) -> list[str]:
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
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
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
        print(f"\nTotal trials: {all_total_trials}")
        print(f"Valid trials: {all_valid_trials}")
        print(f"Rejected trials: {all_rejected_trials}")
        if all_total_trials > 0:
            print(f"Rejection rate: {all_rejected_trials / all_total_trials * 100:.1f}%")
        print("=" * 50)

        return output_paths


def build_eegeyenet(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build EEGEyeNet dataset.

    Args:
        raw_data_dir: Directory containing BIDS format raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for EEGEyeNetBuilder

    Returns:
        List of output file paths
    """
    builder = EEGEyeNetBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build EEGEyeNet HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing BIDS format raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling frequency")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=2.0, help="Stride length in seconds")
    args = parser.parse_args()

    build_eegeyenet(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
