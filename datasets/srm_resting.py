"""
SRM Resting-state EEG Dataset Builder.

SRM Resting-state EEG Dataset
- 111 subjects
- 64 channels (10-10 system)
- 1024 Hz sampling rate
- 240 seconds (4 minutes) recording per session
- Multiple sessions per subject (ses-t1, ses-t2)
- Resting state eyes closed task
- https://openneuro.org/datasets/ds003775

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
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


# SRM Resting-state EEG Dataset Configuration
SRM_RESTING_INFO = DatasetInfo(
    dataset_name="SRM_Resting_EEG",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=1,  # Resting state has no class labels, but we need num_labels > 0
    category_list=["resting"],  # Single category for resting state
    sampling_rate=200.0,  # Target sampling rate (downsampled from 1024 Hz)
    montage="10_10",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.

    MNE uses Volts internally, so we need to ensure data is in V before processing.
    This function detects the unit based on amplitude range and converts accordingly.
    
    Uses robust statistics (percentile) instead of max to avoid noise/artifact interference.

    Args:
        data: Input data array (shape: n_channels x n_samples)

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    # Use 99th percentile instead of max to be robust against noise/artifacts
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    
    # Fallback to max if percentile is too small (edge case)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    # Also check median absolute deviation (MAD) as a sanity check
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


class SRMRestingBuilder:
    """Builder for SRM Resting-state EEG dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50 Hz for Europe
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize SRM Resting-state EEG builder.

        Args:
            raw_data_dir: Directory containing raw files (ds003775-download)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50 Hz for Europe)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "SRM_Resting":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "SRM_Resting"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1024.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-111, but some may be missing)."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            try:
                sub_id = int(sub_dir.name.split("-")[1])
                # Check if any session has EDF file
                for ses_dir in sub_dir.glob("ses-*"):
                    eeg_dir = ses_dir / "eeg"
                    edf_files = list(eeg_dir.glob("*_task-resteyesc_eeg.edf"))
                    if edf_files:
                        subject_ids.append(sub_id)
                        break
            except (ValueError, IndexError):
                continue
        return sorted(subject_ids)

    def _find_files(self, subject_id: int) -> list[dict]:
        """
        Find all EDF files for a subject across all sessions.
        
        Returns:
            List of dicts with keys: 'edf', 'session_id'
        """
        sub_dir = self.raw_data_dir / f"sub-{subject_id:03d}"
        files = []
        
        # Find all sessions
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses_name = ses_dir.name  # e.g., "ses-t1"
            ses_id = int(ses_name.split("-")[1][1:])  # Extract number from "t1" -> 1
            
            eeg_dir = ses_dir / "eeg"
            edf_file = eeg_dir / f"sub-{subject_id:03d}_{ses_name}_task-resteyesc_eeg.edf"
            
            if edf_file.exists():
                files.append({
                    'edf': edf_file,
                    'session_id': ses_id,
                    'session_name': ses_name,
                })
        
        if not files:
            raise FileNotFoundError(f"No EDF files found for subject {subject_id}")
        
        return files

    def _read_raw(self, edf_file: Path):
        """
        Read EDF file and convert to MNE Raw object.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EDF files")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        
        # MNE's EDF reader typically returns data in Volts already
        # Check if data is in reasonable range for Volts (0.001 to 1 V)
        # If data is much larger (> 10 V), it might be in µV and needs conversion
        if hasattr(raw, '_data') and raw._data is not None:
            max_amp = np.abs(raw._data).max()
            # If max amplitude > 10 V, likely in µV (typical EEG is 10-100 µV = 1e-5 to 1e-4 V)
            if max_amp > 10.0:
                raw._data = raw._data / 1e6
                print(f"  Detected unit: µV (max={max_amp:.2e}), converted to V")
            elif max_amp > 1.0:
                # Between 1-10 V, could be mV
                raw._data = raw._data / 1e3
                print(f"  Detected unit: mV (max={max_amp:.2e}), converted to V")
            # Otherwise, assume already in Volts (typical range: 0.001-1 V)
        
        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter (50 Hz for Europe)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """
        Validate trial amplitude.

        Args:
            trial_data: Trial data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_trials}")
        print(f"  Valid segments: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building SRM Resting-state EEG dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        
        all_segments = []
        ch_names = None
        trial_counter = 0
        session_time_offset = 0.0

        # Process each session
        for file_info in files:
            edf_file = file_info['edf']
            session_id = file_info['session_id']
            
            print(f"Reading {edf_file}")
            raw = self._read_raw(edf_file)
            raw = self._preprocess(raw)
            
            if ch_names is None:
                ch_names = raw.ch_names
            
            # Get data (in Volts)
            data = raw.get_data()  # shape: (n_channels, n_samples)
            n_channels, n_samples = data.shape
            
            # Segment into windows (sliding window for resting state)
            for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end_sample = start_sample + self.window_samples
                segment_data_v = data[:, start_sample:end_sample]
                segment_data_uv = segment_data_v * 1e6  # Convert V to µV
                
                # Validate segment amplitude
                self.total_trials += 1
                if not self._validate_trial(segment_data_uv):
                    max_amp = np.abs(segment_data_uv).max()
                    self.rejected_trials += 1
                    continue
                
                self.valid_trials += 1
                
                # Calculate absolute time within recording
                start_time = session_time_offset + start_sample / self.target_sfreq
                end_time = start_time + self.window_sec
                
                all_segments.append({
                    'data': segment_data_uv,  # Store in µV
                    'trial_id': trial_counter,
                    'session_id': session_id,
                    'start_time': start_time,
                    'end_time': end_time,
                })
                trial_counter += 1
            
            # Update session time offset
            session_time_offset += n_samples / self.target_sfreq

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SRM_RESTING_INFO.dataset_name,
            task_type=SRM_RESTING_INFO.task_type.value,
            downstream_task_type=SRM_RESTING_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SRM_RESTING_INFO.num_labels,
            category_list=SRM_RESTING_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_10",
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
                    segment_id=0,  # Single segment per trial for resting state
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=np.array([0]),  # Resting state has no class label, use 0
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": SRM_RESTING_INFO.dataset_name,
                "description": "SRM Resting-state EEG Dataset",
                "task_type": str(SRM_RESTING_INFO.task_type.value),
                "downstream_task": str(SRM_RESTING_INFO.downstream_task_type.value),
                "num_labels": SRM_RESTING_INFO.num_labels,
                "category_list": SRM_RESTING_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": SRM_RESTING_INFO.channels,
                "montage": SRM_RESTING_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds003775",
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
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_trials
                all_valid_segments += self.valid_trials
                all_rejected_segments += self.rejected_trials
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


def build_srm_resting(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build SRM Resting-state EEG dataset.

    Args:
        raw_data_dir: Directory containing raw files (ds003775-download)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for SRMRestingBuilder

    Returns:
        List of output file paths
    """
    builder = SRMRestingBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SRM Resting-state EEG HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (ds003775-download)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_srm_resting(args.raw_data_dir, args.output_dir, args.subjects)

