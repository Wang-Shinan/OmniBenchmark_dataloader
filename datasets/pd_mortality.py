"""
Parkinson's Disease Mortality Dataset Builder.

EEG Mortality Dataset in Parkinson's Disease
- Resting-state EEG recordings from individuals with Parkinson's disease
- Binary classification: living vs deceased
- 500 Hz sampling rate
- 63 EEG channels
- BrainVision format (.vhdr, .eeg, .vmrk)
- https://openneuro.org/datasets/ds007020

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


# PD Mortality Dataset Configuration
PD_MORTALITY_INFO = DatasetInfo(
    dataset_name="PD_Mortality",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["living", "deceased"],  # 0=living, 1=deceased
    sampling_rate=200.0,  # Target sampling rate (downsampled from 500 Hz)
    montage="10_20",
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Label mapping: survival_status -> class index
LABEL_MAP = {
    'living': 0,
    'deceased': 1,
}

# Default amplitude threshold (µV) for validation
DEFAULT_MAX_AMPLITUDE_UV = 600.0


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


class PDMortalityBuilder:
    """Builder for Parkinson's Disease Mortality dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 60.0,  # 60 Hz for USA
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize PD Mortality builder.

        Args:
            raw_data_dir: Directory containing raw files (BIDS format)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (60 Hz for USA)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "PD_Mortality":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "PD_Mortality"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 500.0  # Original sampling rate
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
        
        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = None
        
        # Load participants metadata
        self._load_participants()

    def _load_participants(self):
        """Load participants.tsv to get survival status labels."""
        participants_file = self.raw_data_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found: {participants_file}")
        
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        # Ensure participant_id is string
        self.participants_df['participant_id'] = self.participants_df['participant_id'].astype(str).str.strip()
        # Create label mapping
        self.participant_labels = {}
        for _, row in self.participants_df.iterrows():
            participant_id = row['participant_id']
            survival_status = row['survival_status'].strip().lower()
            if survival_status in LABEL_MAP:
                self.participant_labels[participant_id] = LABEL_MAP[survival_status]
            else:
                print(f"  Warning: Unknown survival_status '{survival_status}' for {participant_id}")

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs (BIDS format: sub-PD####)."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            subject_id = sub_dir.name
            # Check if EEG file exists
            vhdr_file = sub_dir / "ses-01" / "eeg" / f"{subject_id}_ses-01_task-rest_eeg.vhdr"
            if vhdr_file.exists() and subject_id in self.participant_labels:
                subject_ids.append(subject_id)
        return sorted(subject_ids)

    def _find_files(self, subject_id: str) -> Path:
        """
        Find EEG file for a subject.
        
        Returns:
            Path to vhdr file
        """
        vhdr_file = self.raw_data_dir / subject_id / "ses-01" / "eeg" / f"{subject_id}_ses-01_task-rest_eeg.vhdr"
        
        if not vhdr_file.exists():
            raise FileNotFoundError(f"EEG vhdr file not found for subject {subject_id}: {vhdr_file}")
        
        return vhdr_file

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
        # Notch filter (60 Hz for USA)
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
            subject_id: Subject identifier (BIDS format: sub-PD####)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building PD Mortality dataset")

        # Reset validation counters
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        # Get label for this subject
        if subject_id not in self.participant_labels:
            raise ValueError(f"No label found for subject {subject_id} in participants.tsv")
        label = self.participant_labels[subject_id]

        # Find files
        vhdr_file = self._find_files(subject_id)
        
        # Read raw data
        print(f"Reading {vhdr_file.name}")
        raw = self._read_raw(vhdr_file)
        if raw is None:
            raise ValueError(f"Failed to read raw data for subject {subject_id}")
        
        raw = self._preprocess(raw)
        
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
        all_segments = []
        for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
            end_sample = start_sample + self.window_samples
            
            # Ensure we don't exceed data bounds
            if end_sample > n_samples:
                break
            
            segment_data_uv = data_uv[:, start_sample:end_sample]
            
            # Validate segment shape (must match window_samples exactly)
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
                'data': segment_data_uv,  # Store in µV, shape: (n_channels, window_samples)
                'start_time': start_time,
                'end_time': end_time,
            })
        
        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # Create subject attributes
        # Extract numeric ID from subject_id (e.g., "sub-PD1001" -> 1001)
        try:
            numeric_id = int(subject_id.split("-")[1].replace("PD", ""))
        except (ValueError, IndexError):
            numeric_id = hash(subject_id) % 100000  # Fallback to hash
        
        subject_attrs = SubjectAttrs(
            subject_id=numeric_id,
            dataset_name=PD_MORTALITY_INFO.dataset_name,
            task_type=PD_MORTALITY_INFO.task_type.value,
            downstream_task_type=PD_MORTALITY_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=PD_MORTALITY_INFO.num_labels,
            category_list=PD_MORTALITY_INFO.category_list,
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
        output_path = self.output_dir / f"{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            # Single trial with multiple segments
            trial_attrs = TrialAttrs(
                trial_id=0,
                session_id=1,
            )
            trial_name = writer.add_trial(trial_attrs)

            label_arr = np.array([label], dtype=np.int64)

            for seg_id, segment in enumerate(all_segments):
                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=label_arr,
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available, otherwise use from config
        channels = self._dataset_channels if self._dataset_channels else PD_MORTALITY_INFO.channels
        
        info = {
            "dataset": {
                "name": PD_MORTALITY_INFO.dataset_name,
                "description": "EEG Mortality Dataset in Parkinson's Disease - Resting-state EEG for mortality prediction",
                "task_type": str(PD_MORTALITY_INFO.task_type.value),
                "downstream_task": str(PD_MORTALITY_INFO.downstream_task_type.value),
                "num_labels": PD_MORTALITY_INFO.num_labels,
                "category_list": PD_MORTALITY_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": PD_MORTALITY_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds007020",
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

    def build_all(self, subject_ids: list[str] = None) -> list[str]:
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


def build_pd_mortality(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build PD Mortality dataset.

    Args:
        raw_data_dir: Directory containing raw files (BIDS format)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for PDMortalityBuilder

    Returns:
        List of output file paths
    """
    builder = PDMortalityBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PD Mortality HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (BIDS format)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=str, help="Subject IDs to process (e.g., sub-PD1001)")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate (default: 200.0 Hz)")
    parser.add_argument("--window_sec", type=float, default=1.0, help="Window length in seconds (default: 1.0)")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Stride length in seconds (default: 1.0)")
    parser.add_argument("--filter_low", type=float, default=0.1, help="Low cutoff frequency (default: 0.1 Hz)")
    parser.add_argument("--filter_high", type=float, default=75.0, help="High cutoff frequency (default: 75.0 Hz)")
    parser.add_argument("--filter_notch", type=float, default=60.0, help="Notch filter frequency (default: 60.0 Hz)")
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV, help=f"Amplitude threshold in µV (default: {DEFAULT_MAX_AMPLITUDE_UV})")
    args = parser.parse_args()

    build_pd_mortality(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )
