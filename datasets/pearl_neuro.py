"""
PEARL-Neuro Dataset Builder.

PEARL-Neuro: A Polish Electroencephalography, Alzheimer's Risk-genes, Lifestyle and Neuroimaging Database
- 79 subjects
- 3 tasks: rest, msit (cognitive control), sternberg (working memory)
- 127 channels
- 1000 Hz sampling rate
- BrainVision format (.vhdr, .eeg, .vmrk)
- https://openneuro.org/datasets/ds004796

MSIT Task (Multi-Source Interference Task):
- Cognitive control task
- 2 classes: FS condition (S 4) vs 00 condition (S 5)
- Trials extracted from stimulus events

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


# PEARL-Neuro MSIT Dataset Configuration
PEARL_MSIT_INFO = DatasetInfo(
    dataset_name="PEARL_MSIT_2class",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["FS", "00"],  # FS condition vs 00 condition
    sampling_rate=200.0,  # Target sampling rate (downsampled from 1000 Hz)
    montage="10_10",  # 127-channel system
    channels=[],  # Will be populated from Raw.ch_names at runtime
)

# Label mapping: event_type -> class index
# S 4 = FS condition, S 5 = 00 condition
PEARL_MSIT_LABEL_MAP = {
    'S  4': 0,  # FS condition
    'S  5': 1,  # 00 condition
}

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


class PEARLMSITBuilder:
    """Builder for PEARL-Neuro MSIT dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,  # MSIT trials are typically short
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # Poland uses 50 Hz
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize PEARL-Neuro MSIT builder.

        Args:
            raw_data_dir: Directory containing raw files (ds004796-download)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50 Hz for Poland)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "PEARL_MSIT":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "PEARL_MSIT"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0  # Original sampling rate
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
        
        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = None

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-80, but some may be missing)."""
        subject_dirs = sorted(self.raw_data_dir.glob("sub-*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            try:
                sub_id = int(sub_dir.name.split("-")[1])
                # Check if MSIT task file exists
                msit_file = sub_dir / "eeg" / f"{sub_dir.name}_task-msit_eeg.vhdr"
                if msit_file.exists():
                    subject_ids.append(sub_id)
            except (ValueError, IndexError):
                continue
        return sorted(subject_ids)

    def _find_files(self, subject_id: int) -> dict[str, Path]:
        """
        Find MSIT task files for a subject.
        
        Returns:
            dict with keys: 'vhdr', 'events'
        """
        sub_dir = self.raw_data_dir / f"sub-{subject_id:02d}"
        eeg_dir = sub_dir / "eeg"
        
        vhdr_file = eeg_dir / f"sub-{subject_id:02d}_task-msit_eeg.vhdr"
        events_file = eeg_dir / f"sub-{subject_id:02d}_task-msit_events.tsv"
        
        if not vhdr_file.exists():
            raise FileNotFoundError(f"MSIT vhdr file not found for subject {subject_id}: {vhdr_file}")
        if not events_file.exists():
            raise FileNotFoundError(f"MSIT events file not found for subject {subject_id}: {events_file}")
        
        return {
            'vhdr': vhdr_file,
            'events': events_file,
        }

    def _load_events_tsv(self, events_file: Path) -> pd.DataFrame:
        """Load events from TSV file."""
        events = pd.read_csv(events_file, sep='\t')
        # Note: According to task-msit_events.json, onset is in milliseconds,
        # but the actual values in the TSV appear to be in seconds already.
        # Check if conversion is needed by examining the first value
        if 'onset' in events.columns and len(events) > 0:
            first_onset = events['onset'].iloc[0]
            # If first onset is > 1000, it's likely in milliseconds, convert to seconds
            if first_onset > 1000:
                events['onset'] = events['onset'] / 1000.0
        return events

    def _read_raw(self, vhdr_file: Path):
        """
        Read BrainVision file and convert to MNE Raw object.
        
        Note: MNE's BrainVision reader typically handles unit conversion automatically,
        so we trust MNE's unit handling and don't perform additional conversion.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for reading BrainVision files")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=True, verbose=False)
        
        # MNE's BrainVision reader typically returns data in Volts already
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
        # Notch filter (50 Hz for Poland)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _extract_trials(self, raw, events_df: pd.DataFrame) -> list[dict]:
        """
        Extract trials from events DataFrame.
        
        Args:
            raw: MNE Raw object
            events_df: DataFrame with columns: onset, event_type, trial_type
        
        Returns:
            List of trial dicts with keys: 'onset', 'label'
        """
        trials = []
        
        # Filter for stimulus events with valid labels
        stimulus_events = events_df[
            (events_df['trial_type'] == 'stimulus') & 
            (events_df['event_type'].isin(PEARL_MSIT_LABEL_MAP.keys()))
        ]
        
        for _, event in stimulus_events.iterrows():
            onset = event['onset']
            event_type = event['event_type'].strip()  # Remove extra spaces
            
            if event_type in PEARL_MSIT_LABEL_MAP:
                label = PEARL_MSIT_LABEL_MAP[event_type]
                trials.append({
                    'onset': onset,
                    'label': label,
                })
        
        return trials

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
        """Report trial validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building PEARL-Neuro MSIT dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        
        # Load events
        events_df = self._load_events_tsv(files['events'])
        
        # Read raw data
        print(f"Reading {files['vhdr']}")
        raw = self._read_raw(files['vhdr'])
        raw = self._preprocess(raw)
        
        ch_names = raw.ch_names
        
        # Store channel names for dataset info (first subject sets it)
        if not hasattr(self, '_dataset_channels') or self._dataset_channels is None:
            self._dataset_channels = ch_names
        
        # Extract trials
        trials = self._extract_trials(raw, events_df)
        data = raw.get_data()  # In Volts
        
        all_trials = []
        for trial_idx, trial in enumerate(trials):
            onset_sample = int(trial['onset'] * self.target_sfreq)
            end_sample = onset_sample + self.window_samples

            if end_sample <= data.shape[1]:
                # Data from MNE is in Volts (V), convert to microvolts (µV)
                trial_data_v = data[:, onset_sample:end_sample]
                trial_data_uv = trial_data_v * 1e6  # Convert V to µV

                # Validate trial amplitude
                self.total_trials += 1
                if not self._validate_trial(trial_data_uv):
                    max_amp = np.abs(trial_data_uv).max()
                    print(f"  Skipping trial {trial_idx}: amplitude {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                    self.rejected_trials += 1
                    continue

                self.valid_trials += 1
                all_trials.append({
                    'data': trial_data_uv,  # Store in µV
                    'label': trial['label'],
                    'trial_id': len(all_trials),
                    'onset_time': trial['onset'],
                })

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=PEARL_MSIT_INFO.dataset_name,
            task_type=PEARL_MSIT_INFO.task_type.value,
            downstream_task_type=PEARL_MSIT_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=PEARL_MSIT_INFO.num_labels,
            category_list=PEARL_MSIT_INFO.category_list,
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
            for trial in all_trials:
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=1,  # Single session per subject
                )
                trial_name = writer.add_trial(trial_attrs)

                # Single segment per trial
                start_time = trial['onset_time']
                end_time = start_time + self.window_sec

                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=start_time,
                    end_time=end_time,
                    time_length=self.window_sec,
                    label=np.array([trial['label']]),
                )
                writer.add_segment(trial_name, segment_attrs, trial['data'])

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available, otherwise use from config
        channels = self._dataset_channels if hasattr(self, '_dataset_channels') and self._dataset_channels else PEARL_MSIT_INFO.channels
        
        info = {
            "dataset": {
                "name": PEARL_MSIT_INFO.dataset_name,
                "description": "PEARL-Neuro MSIT Task: Cognitive Control",
                "task_type": str(PEARL_MSIT_INFO.task_type.value),
                "downstream_task": str(PEARL_MSIT_INFO.downstream_task_type.value),
                "num_labels": PEARL_MSIT_INFO.num_labels,
                "category_list": PEARL_MSIT_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "channels": channels,
                "montage": PEARL_MSIT_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds004796",
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

        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_trials": all_total_trials,
            "valid_trials": all_valid_trials,
            "rejected_trials": all_rejected_trials,
            "rejection_rate": f"{all_rejected_trials / all_total_trials * 100:.1f}%" if all_total_trials > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_pearl_msit(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build PEARL-Neuro MSIT dataset.

    Args:
        raw_data_dir: Directory containing raw files (ds004796-download)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for PEARLMSITBuilder

    Returns:
        List of output file paths
    """
    builder = PEARLMSITBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PEARL-Neuro MSIT HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (ds004796-download)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    args = parser.parse_args()

    build_pearl_msit(args.raw_data_dir, args.output_dir, args.subjects)

