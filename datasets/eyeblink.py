"""
Eyeblink Dataset Builder.

Eyeblink Dataset: Voluntary vs Involuntary Blink Classification
- 2 classes: Voluntary blink, Involuntary blink
- 15 channels
- Epochs: 1024 samples per epoch
- Data from Epochs.mat (pre-epoched data)
- Raw data also available in RawData.mat (20 subjects, 3 trials each)

Task: Binary classification (Voluntary vs Involuntary blink detection)

Data Unit Handling:
- Data appears to be in microvolts (µV) based on amplitude range (-494 to 318)
- Convert to Volts for MNE processing, then back to µV for HDF5 storage
- Default amplitude validation threshold: 600 µV
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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


# Eyeblink Dataset Configuration
EYEBLINK_INFO = DatasetInfo(
    dataset_name="Eyeblink",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["Involuntary", "Voluntary"],  # 0=Involuntary, 1=Voluntary
    sampling_rate=256.0,  # Estimated based on epoch length (1024 samples ≈ 4s at 256Hz)
    montage="10_20",
    channels=[],  # Will be populated from data (15 channels)
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Estimated epoch duration (seconds) based on 1024 samples at 256Hz
EPOCH_DURATION_SEC = 1024 / 256.0  # ≈ 4 seconds


def detect_unit_and_convert_to_volts(data: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.
    
    Uses robust statistics (percentile) instead of max to avoid noise/artifact interference.
    
    Args:
        data: Input data array (shape: n_channels x n_samples or n_epochs x n_channels x n_samples)
    
    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    # Flatten for percentile calculation
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


class EyeblinkBuilder:
    """Builder for Eyeblink dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz or 60Hz depending on location
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        use_epochs: bool = True,  # Use pre-epoched data from Epochs.mat
    ):
        """
        Initialize Eyeblink builder.

        Args:
            raw_data_dir: Directory containing Epochs.mat and RawData.mat
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency (default: 256 Hz)
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50 Hz or 60 Hz)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            use_epochs: If True, use pre-epoched data from Epochs.mat; else use RawData.mat
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "Eyeblink":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "Eyeblink"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 256.0  # Estimated original sampling rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.use_epochs = use_epochs

        # Track validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Store actual channels from data (will be set during processing)
        self._dataset_channels = None

    def _read_epochs_mat(self, epochs_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read epochs from Epochs.mat file.
        
        Returns:
            tuple: (involuntary_epochs, voluntary_epochs)
            Each is shape (n_epochs, n_channels, n_samples)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")
        
        epochs_data = scipy.io.loadmat(str(epochs_file), struct_as_record=False, squeeze_me=False)
        all_epochs = epochs_data['Epochs'][0, 0].AllEpochs[0, 0]
        
        involuntary = all_epochs.Involuntary  # shape: (n_epochs, n_channels, n_samples)
        voluntary = all_epochs.Voluntary
        
        # Convert from (n_epochs, n_channels, n_samples) to (n_channels, n_samples) per epoch
        # But we'll process epoch by epoch, so keep the shape
        
        return involuntary, voluntary

    def _preprocess_epoch(self, epoch_data: np.ndarray) -> np.ndarray:
        """
        Preprocess a single epoch.
        
        Args:
            epoch_data: Epoch data (n_channels x n_samples) in µV
        
        Returns:
            Preprocessed epoch data (n_channels x n_samples) in µV
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for preprocessing")
        
        # Convert to Volts for MNE
        epoch_v, _ = detect_unit_and_convert_to_volts(epoch_data)
        
        # Create MNE Raw object
        n_channels, n_samples = epoch_v.shape
        ch_names = [f"CH{i+1}" for i in range(n_channels)]
        
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.orig_sfreq,
            ch_types=['eeg'] * n_channels
        )
        
        raw = mne.io.RawArray(epoch_v, info, verbose=False)
        
        # Apply preprocessing
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        
        # Get processed data and convert back to µV
        processed_data = raw.get_data() * 1e6  # Convert V to µV
        
        return processed_data

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """
        Validate trial amplitude.

        Args:
            trial_data: Trial data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def build_all(self) -> List[str]:
        """
        Build HDF5 files for all epochs.

        Returns:
            List of output file paths (one file per subject, or single file for all epochs)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for building Eyeblink dataset")
        
        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find Epochs.mat file
        epochs_file = self.raw_data_dir / "Epochs.mat"
        if not epochs_file.exists():
            # Try subdirectory
            epochs_file = self.raw_data_dir / "EyeblinkDataset" / "Epochs.mat"
            if not epochs_file.exists():
                raise FileNotFoundError(f"Epochs.mat not found in {self.raw_data_dir}")
        
        print(f"Reading {epochs_file}")
        involuntary_epochs, voluntary_epochs = self._read_epochs_mat(epochs_file)
        
        print(f"Found {involuntary_epochs.shape[0]} involuntary epochs")
        print(f"Found {voluntary_epochs.shape[0]} voluntary epochs")
        
        # Process epochs
        all_trials = []
        
        # Process involuntary epochs (label 0)
        for epoch_idx in range(involuntary_epochs.shape[0]):
            epoch_data = involuntary_epochs[epoch_idx]  # Shape: (n_channels, n_samples) = (15, 1024)
            
            # Preprocess
            try:
                processed_epoch = self._preprocess_epoch(epoch_data)
            except Exception as e:
                print(f"  Warning: Failed to preprocess involuntary epoch {epoch_idx}: {e}")
                continue
            
            # Validate
            self.total_trials += 1
            if not self._validate_trial(processed_epoch):
                self.rejected_trials += 1
                continue
            
            self.valid_trials += 1
            
            # Store channel names
            if self._dataset_channels is None:
                n_channels = processed_epoch.shape[0]
                self._dataset_channels = [f"CH{i+1}" for i in range(n_channels)]
            
            all_trials.append({
                'data': processed_epoch,
                'label': 0,  # Involuntary
                'trial_id': len(all_trials),
            })
        
        # Process voluntary epochs (label 1)
        for epoch_idx in range(voluntary_epochs.shape[0]):
            epoch_data = voluntary_epochs[epoch_idx]  # Shape: (n_channels, n_samples) = (15, 1024)
            
            # Preprocess
            try:
                processed_epoch = self._preprocess_epoch(epoch_data)
            except Exception as e:
                print(f"  Warning: Failed to preprocess voluntary epoch {epoch_idx}: {e}")
                continue
            
            # Validate
            self.total_trials += 1
            if not self._validate_trial(processed_epoch):
                self.rejected_trials += 1
                continue
            
            self.valid_trials += 1
            
            all_trials.append({
                'data': processed_epoch,
                'label': 1,  # Voluntary
                'trial_id': len(all_trials),
            })
        
        if not all_trials:
            raise ValueError("No valid trials extracted")
        
        # Create a single HDF5 file with all epochs
        # Since we don't have subject information in Epochs.mat, we'll create one file
        # with all epochs as separate trials
        
        # Use default channel names if not set
        if self._dataset_channels is None:
            self._dataset_channels = [f"CH{i+1}" for i in range(15)]
        
        # Create subject attributes (using a dummy subject ID)
        subject_attrs = SubjectAttrs(
            subject_id="all_subjects",
            dataset_name=EYEBLINK_INFO.dataset_name,
            task_type=EYEBLINK_INFO.task_type.value,
            downstream_task_type=EYEBLINK_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=self._dataset_channels,
            num_labels=EYEBLINK_INFO.num_labels,
            category_list=EYEBLINK_INFO.category_list,
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
        output_path = self.output_dir / "all_epochs.h5"

        # Calculate epoch duration
        epoch_duration = all_trials[0]['data'].shape[1] / self.target_sfreq

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=1,
                )
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=0.0,
                    end_time=epoch_duration,
                    time_length=epoch_duration,
                    label=np.array([trial['label']]),
                )
                writer.add_segment(trial_name, segment_attrs, trial['data'])

        # Report validation statistics
        self._report_validation_stats()
        print(f"Saved {output_path} ({self.valid_trials} valid trials)")

        # Save dataset info JSON
        self._save_dataset_info()

        return [str(output_path)]

    def _save_dataset_info(self):
        """Save dataset info and processing parameters to JSON."""
        channels = self._dataset_channels if self._dataset_channels else EYEBLINK_INFO.channels
        
        info = {
            "dataset": {
                "name": EYEBLINK_INFO.dataset_name,
                "description": "Eyeblink Dataset - Voluntary vs Involuntary Blink Classification",
                "task_type": str(EYEBLINK_INFO.task_type.value),
                "downstream_task": str(EYEBLINK_INFO.downstream_task_type.value),
                "num_labels": EYEBLINK_INFO.num_labels,
                "category_list": EYEBLINK_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": EYEBLINK_INFO.montage,
                "source_url": "EyeblinkDataset",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "use_epochs": self.use_epochs,
            },
            "statistics": {
                "total_trials": self.total_trials,
                "valid_trials": self.valid_trials,
                "rejected_trials": self.rejected_trials,
                "rejection_rate": f"{self.rejected_trials / self.total_trials * 100:.1f}%" if self.total_trials > 0 else "0%",
            },
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


def build_eyeblink(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    **kwargs,
) -> List[str]:
    """
    Convenience function to build Eyeblink dataset.

    Args:
        raw_data_dir: Directory containing Epochs.mat and RawData.mat
        output_dir: Output directory for HDF5 files
        **kwargs: Additional arguments for EyeblinkBuilder

    Returns:
        List of output file paths
    """
    builder = EyeblinkBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Eyeblink HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing Epochs.mat and RawData.mat")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--target_sfreq", type=float, default=256.0, help="Target sampling rate (default: 256.0 Hz)")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency (default: 50.0 Hz)")
    args = parser.parse_args()

    build_eyeblink(args.raw_data_dir, args.output_dir, target_sfreq=args.target_sfreq, filter_notch=args.filter_notch)
