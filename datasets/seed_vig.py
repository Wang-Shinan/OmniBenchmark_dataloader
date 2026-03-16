"""
SEED-VIG Dataset Builder.

SEED-VIG: A Multimodal Dataset with EEG and forehead EOG for Vigilance Estimation
- 23 experiments (sessions)
- 17 EEG channels
- 200 Hz sampling rate
- Continuous vigilance labels (PERCLOS: 0-1, awake to drowsy)
- 885 samples per experiment
- Each sample corresponds to ~8 seconds of EEG data
- https://bcmi.sjtu.edu.cn/~seed/

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)

Note: This is a regression task (continuous labels), not classification.
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import mne
    from scipy.io import loadmat
    HAS_MNE = True
    HAS_SCIPY = True
except ImportError:
    HAS_MNE = False
    HAS_SCIPY = False

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


# SEED-VIG Dataset Configuration
SEED_VIG_INFO = DatasetInfo(
    dataset_name="SEED_VIG",
    task_type=DatasetTaskType.RESTING_STATE,  # Vigilance estimation during resting/driving
    downstream_task_type=DownstreamTaskType.REGRESSION,  # Continuous labels (0-1)
    num_labels=1,  # Regression task (single continuous value)
    category_list=[],  # Not applicable for regression
    sampling_rate=200.0,  # Original sampling rate
    montage="10_20",
    channels=[],  # Will be populated from data
)

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0

# Expected channels (17 channels from temporal and posterior brain areas)
EXPECTED_CHANNELS = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 
                     'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']


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


class SEEDVIGBuilder:
    """Builder for SEED-VIG dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        label_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,  # Keep original sampling rate
        window_sec: float = 8.0,  # Each sample is ~8 seconds
        stride_sec: float = 8.0,  # Non-overlapping windows
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50 Hz for China
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize SEED-VIG builder.

        Args:
            raw_data_dir: Directory containing raw EEG .mat files
            label_dir: Directory containing PERCLOS label .mat files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency (default: 200 Hz, same as original)
            window_sec: Window length in seconds (default: 8.0, matching sample duration)
            stride_sec: Stride length in seconds (default: 8.0, non-overlapping)
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50 Hz for China)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.label_dir = Path(label_dir)
        output_path = Path(output_dir)
        if output_path.name == "SEED_VIG":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "SEED_VIG"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 200.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        
        # Store actual channels from data (will be set during first experiment processing)
        self._dataset_channels = None

    def get_experiment_ids(self) -> list[str]:
        """Get list of experiment IDs (file names without .mat extension)."""
        raw_files = sorted(self.raw_data_dir.glob("*.mat"))
        label_files = sorted(self.label_dir.glob("*.mat"))
        
        # Get intersection of files that exist in both directories
        raw_ids = {f.stem for f in raw_files}
        label_ids = {f.stem for f in label_files}
        experiment_ids = sorted(raw_ids & label_ids)
        
        return experiment_ids

    def _find_files(self, experiment_id: str) -> dict[str, Path]:
        """
        Find EEG and label files for an experiment.
        
        Returns:
            dict with keys: 'eeg', 'label'
        """
        eeg_file = self.raw_data_dir / f"{experiment_id}.mat"
        label_file = self.label_dir / f"{experiment_id}.mat"
        
        if not eeg_file.exists():
            raise FileNotFoundError(f"EEG file not found for experiment {experiment_id}: {eeg_file}")
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found for experiment {experiment_id}: {label_file}")
        
        return {
            'eeg': eeg_file,
            'label': label_file,
        }

    def _read_raw_mat(self, eeg_file: Path):
        """
        Read .mat file and convert to MNE Raw object.
        
        SEED-VIG .mat file structure:
        - EEG: struct with fields:
          - data: EEG data (n_samples x n_channels)
          - chn: channel names
          - sample_rate: sampling rate
          - node_number: number of channels
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")
        
        try:
            mat_data = loadmat(str(eeg_file), struct_as_record=False, squeeze_me=True)
        except Exception as e:
            raise ValueError(f"Failed to load .mat file {eeg_file}: {e}") from e
        
        if 'EEG' not in mat_data:
            raise ValueError(f"EEG field not found in {eeg_file}")
        
        eeg_struct = mat_data['EEG']
        
        # Extract data and metadata
        data = eeg_struct.data  # Shape: (n_samples, n_channels)
        ch_names = [str(ch) for ch in eeg_struct.chn] if hasattr(eeg_struct, 'chn') else EXPECTED_CHANNELS
        fs = float(eeg_struct.sample_rate) if hasattr(eeg_struct, 'sample_rate') else self.orig_sfreq
        
        # Transpose to (n_channels, n_samples) for MNE
        data = data.T.astype(np.float64)
        
        # Auto-detect unit and convert to Volts
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        print(f"  Detected unit: {detected_unit}, max amplitude: {np.abs(data).max():.2e}")
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=['eeg'] * len(ch_names)
        )
        
        # Create MNE Raw object
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        
        return raw

    def _load_labels(self, label_file: Path) -> np.ndarray:
        """Load PERCLOS labels from .mat file."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")
        
        try:
            mat_data = loadmat(str(label_file), struct_as_record=False, squeeze_me=True)
        except Exception as e:
            raise ValueError(f"Failed to load label file {label_file}: {e}") from e
        
        if 'perclos' not in mat_data:
            raise ValueError(f"perclos field not found in {label_file}")
        
        perclos = mat_data['perclos']
        # Ensure it's 1D array
        if perclos.ndim > 1:
            perclos = perclos.flatten()
        
        return perclos.astype(np.float32)

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter (50 Hz for China)
        if self.filter_notch > 0:
            nyquist = raw.info['sfreq'] / 2.0
            if self.filter_notch < nyquist:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            else:
                print(f"  Warning: Skipping notch filter ({self.filter_notch}Hz) - exceeds Nyquist ({nyquist:.1f}Hz)")

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

    def _report_validation_stats(self, experiment_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Experiment {experiment_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def build_experiment(self, experiment_id: str) -> str:
        """
        Build HDF5 file for a single experiment.

        Args:
            experiment_id: Experiment identifier (file name without .mat)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building SEED-VIG dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Find files
        files = self._find_files(experiment_id)
        
        # Load labels
        labels = self._load_labels(files['label'])
        n_labels = len(labels)
        print(f"Loaded {n_labels} PERCLOS labels (range: {labels.min():.3f} - {labels.max():.3f})")
        
        # Read raw data
        print(f"Reading {files['eeg'].name}")
        raw = self._read_raw_mat(files['eeg'])
        raw = self._preprocess(raw)
        
        ch_names = raw.ch_names
        
        # Store channel names for dataset info (first experiment sets it)
        if self._dataset_channels is None:
            self._dataset_channels = ch_names
        
        # Get data (in Volts)
        data = raw.get_data()  # shape: (n_channels, n_samples)
        n_channels, n_samples = data.shape
        
        # Convert to µV
        data_uv = data * 1e6
        
        # Calculate samples per label (should be ~8 seconds = 1600 samples at 200 Hz)
        samples_per_label = n_samples // n_labels
        print(f"  Data: {n_samples} samples, {n_labels} labels, ~{samples_per_label} samples per label")
        
        # Segment data to match labels
        all_segments = []
        for label_idx, label_value in enumerate(labels):
            start_sample = label_idx * samples_per_label
            end_sample = start_sample + self.window_samples
            
            # Ensure we don't exceed data bounds
            if end_sample > n_samples:
                # For the last segment, use available data
                if start_sample < n_samples:
                    end_sample = n_samples
                    # Adjust window_samples if needed
                    actual_window_samples = end_sample - start_sample
                    if actual_window_samples < self.window_samples * 0.5:  # Skip if too short
                        continue
                else:
                    break
            
            segment_data_uv = data_uv[:, start_sample:end_sample]
            
            # Pad if necessary (shouldn't happen, but just in case)
            if segment_data_uv.shape[1] < self.window_samples:
                padding = self.window_samples - segment_data_uv.shape[1]
                segment_data_uv = np.pad(segment_data_uv, ((0, 0), (0, padding)), mode='constant')
            
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
                'data': segment_data_uv,  # Store in µV, shape: (n_channels, window_samples)
                'label': label_value,  # Continuous PERCLOS value (0-1)
                'start_time': start_time,
                'end_time': end_time,
            })
        
        if not all_segments:
            raise ValueError(f"No valid segments extracted for experiment {experiment_id}")

        # Create subject attributes
        # Extract numeric ID from experiment_id (e.g., "1_20151124_noon_2" -> 1)
        try:
            numeric_id = int(experiment_id.split('_')[0])
        except (ValueError, IndexError):
            numeric_id = hash(experiment_id) % 100000  # Fallback to hash
        
        subject_attrs = SubjectAttrs(
            subject_id=numeric_id,
            dataset_name=SEED_VIG_INFO.dataset_name,
            task_type=SEED_VIG_INFO.task_type.value,
            downstream_task_type=SEED_VIG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SEED_VIG_INFO.num_labels,
            category_list=SEED_VIG_INFO.category_list,
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
        output_path = self.output_dir / f"{experiment_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            # Single trial with multiple segments
            trial_attrs = TrialAttrs(
                trial_id=0,
                session_id=numeric_id,  # Use experiment ID as session ID
            )
            trial_name = writer.add_trial(trial_attrs)

            for seg_id, segment in enumerate(all_segments):
                # For regression, label is a single continuous value
                label_arr = np.array([segment['label']], dtype=np.float32)

                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=label_arr,  # Continuous value for regression
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        # Report validation statistics
        self._report_validation_stats(experiment_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available, otherwise use from config
        channels = self._dataset_channels if self._dataset_channels else SEED_VIG_INFO.channels
        
        info = {
            "dataset": {
                "name": SEED_VIG_INFO.dataset_name,
                "description": "SEED-VIG: A Multimodal Dataset with EEG and forehead EOG for Vigilance Estimation",
                "task_type": str(SEED_VIG_INFO.task_type.value),
                "downstream_task": str(SEED_VIG_INFO.downstream_task_type.value),
                "num_labels": SEED_VIG_INFO.num_labels,
                "category_list": SEED_VIG_INFO.category_list,
                "label_type": "continuous",  # PERCLOS values (0-1)
                "label_range": [0.0, 1.0],
                "label_description": "PERCLOS: Percentage of Eyelid Closure (0=awake, 1=drowsy)",
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": SEED_VIG_INFO.montage,
                "source_url": "https://bcmi.sjtu.edu.cn/~seed/",
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

    def build_all(self, experiment_ids: list[str] = None) -> list[str]:
        """
        Build HDF5 files for all experiments.

        Args:
            experiment_ids: List of experiment IDs to process (None = all)

        Returns:
            List of output file paths
        """
        if experiment_ids is None:
            experiment_ids = self.get_experiment_ids()

        output_paths = []
        failed_experiments = []
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for experiment_id in experiment_ids:
            try:
                output_path = self.build_experiment(experiment_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
            except Exception as e:
                print(f"Error processing experiment {experiment_id}: {e}")
                failed_experiments.append(experiment_id)
                import traceback
                traceback.print_exc()

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total experiments: {len(experiment_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_experiments)}")
        if failed_experiments:
            print(f"Failed experiment IDs: {failed_experiments}")
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # Save dataset info JSON
        stats = {
            "total_experiments": len(experiment_ids),
            "successful_experiments": len(output_paths),
            "failed_experiments": failed_experiments,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_seed_vig(
    raw_data_dir: str,
    label_dir: str,
    output_dir: str = "./hdf5",
    experiment_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build SEED-VIG dataset.

    Args:
        raw_data_dir: Directory containing raw EEG .mat files
        label_dir: Directory containing PERCLOS label .mat files
        output_dir: Output directory for HDF5 files
        experiment_ids: List of experiment IDs to process (None = all)
        **kwargs: Additional arguments for SEEDVIGBuilder

    Returns:
        List of output file paths
    """
    builder = SEEDVIGBuilder(raw_data_dir, label_dir, output_dir, **kwargs)
    return builder.build_all(experiment_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SEED-VIG HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw EEG .mat files")
    parser.add_argument("label_dir", help="Directory containing PERCLOS label .mat files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--experiments", nargs="+", type=str, help="Experiment IDs to process")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate (default: 200.0 Hz)")
    parser.add_argument("--window_sec", type=float, default=8.0, help="Window length in seconds (default: 8.0)")
    parser.add_argument("--stride_sec", type=float, default=8.0, help="Stride length in seconds (default: 8.0)")
    parser.add_argument("--filter_low", type=float, default=0.1, help="Low cutoff frequency (default: 0.1 Hz)")
    parser.add_argument("--filter_high", type=float, default=75.0, help="High cutoff frequency (default: 75.0 Hz)")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency (default: 50.0 Hz)")
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV, help=f"Amplitude threshold in µV (default: {DEFAULT_MAX_AMPLITUDE_UV})")
    args = parser.parse_args()

    build_seed_vig(
        args.raw_data_dir,
        args.label_dir,
        args.output_dir,
        args.experiments,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )
