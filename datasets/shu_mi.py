"""
SHU-MI Dataset Builder.

SHU Motor Imagery Dataset (BIDS format).
- 25 subjects (sub-001 to sub-025)
- 5 sessions per subject (ses-01 to ses-05)
- 100 trials per session
- 2 motor imagery classes: left hand, right hand
- BIDS format with .edf/.mat files and .tsv event files
- https://openneuro.org/datasets/ds004504

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)

Data Validation:
- Automatically validate if each trial's amplitude is within reasonable range
- Trials exceeding the threshold will be skipped and recorded
- Validation statistics report will be displayed after processing completes
"""

import os
import json
import csv
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
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
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


# SHU-MI Dataset Configuration
SHU_MI_INFO = DatasetInfo(
    dataset_name="SHU_MI_2class",
    task_type=DatasetTaskType.MOTOR_IMAGINARY,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["left", "right"],
    sampling_rate=1000.0,  # Will be detected from actual data
    montage="10_10",
    channels=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1",
                 "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T3", "T4",
                 "A1", "A2", "CP1", "CP2", "CP5", "CP6", "Pz", "P3",
                 "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"],  # Will be detected from actual data
)

# Label mapping: trial_type -> class index
SHU_MI_LABEL_MAP = {
    'left': 0,
    'right': 1,
}

# Default amplitude threshold (µV)
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


class SHUMIBuilder:
    """Builder for SHU-MI dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz for China
        file_format: str = "auto",  # 'auto', 'edf', 'mat'
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Initialize SHU-MI builder.

        Args:
            raw_data_dir: Directory containing BIDS dataset (should point to the directory with sub-* folders)
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50Hz for China)
            file_format: File format ('auto', 'edf', or 'mat')
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        if output_path.name == "SHU_MI":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "SHU_MI"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = None  # Will be detected from data
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.file_format = file_format
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Track validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-25)."""
        return list(range(1, 26))

    def _detect_file_format(self) -> str:
        """Auto-detect file format from directory structure."""
        if self.file_format != "auto":
            return self.file_format

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.raw_data_dir}")

        # Check for .edf or .mat files directly in raw_data_dir
        edf_files = list(self.raw_data_dir.glob("sub-*_ses-*_task_*_eeg.edf"))
        mat_files = list(self.raw_data_dir.glob("sub-*_ses-*_task_*_eeg.mat"))

        if edf_files:
            return "edf"
        elif mat_files:
            return "mat"

        raise FileNotFoundError("No .edf or .mat files found in dataset")

    def _find_files(self, subject_id: int) -> dict[int, dict]:
        """
        Find all session files for a subject.

        Files are directly in raw_data_dir, not in sub-* subdirectories.
        Format: sub-{subject_id:03d}_ses-{session_id:02d}_task_motorimagery_eeg.{format}

        Args:
            subject_id: Subject identifier (1-25)

        Returns:
            Dictionary mapping session_id to file paths and event file
        """
        files = {}

        if not self.raw_data_dir.exists():
            print(f"⚠️  Warning: Data directory does not exist: {self.raw_data_dir}")
            return files

        file_format = self._detect_file_format()

        # Find all sessions (ses-01 to ses-05)
        for session_id in range(1, 6):
            session_name = f"ses-{session_id:02d}"
            pattern = f"sub-{subject_id:03d}_{session_name}_task_*_eeg.{file_format}"
            data_files = list(self.raw_data_dir.glob(pattern))

            if data_files:
                data_file = data_files[0]
                event_file = self.raw_data_dir / f"sub-{subject_id:03d}_{session_name}_task_motorimagery_events.tsv"

                if event_file.exists():
                    files[session_id] = {
                        'data': data_file,
                        'events': event_file,
                    }
                else:
                    print(f"⚠️  Warning: Event file not found for {data_file}")

        print(f"   Found {len(files)} session(s) for subject {subject_id}")
        return files

    def _read_events_tsv(self, event_file: Path) -> list[dict]:
        """
        Read events from TSV file.

        Args:
            event_file: Path to events.tsv file

        Returns:
            List of event dictionaries with onset, duration, and label
        """
        events = []
        with open(event_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                trial_type = row['trial_type'].strip().lower()
                if trial_type in SHU_MI_LABEL_MAP:
                    events.append({
                        'onset': float(row['onset']) / 1000.0,  # Convert ms to seconds
                        'duration': float(row['duration']) / 1000.0,  # Convert ms to seconds
                        'label': SHU_MI_LABEL_MAP[trial_type],
                    })

        return events

    def _read_raw_edf(self, file_path: Path):
        """Read EDF file and convert to MNE Raw object."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)

            print(f"  Channels: {len(raw.ch_names)}")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  Duration: {raw.times[-1]:.1f} seconds")

            # Auto-detect unit and convert to Volts for MNE processing
            if hasattr(raw, '_data') and raw._data is not None:
                max_amp = np.abs(raw._data).max()
                data_volts, detected_unit = detect_unit_and_convert_to_volts(raw._data)

                if detected_unit != "V":
                    raw._data = data_volts
                    print(f"  Detected unit: {detected_unit}, converted to V (max amplitude: {max_amp:.2e} {detected_unit})")

            return raw

    def _read_raw_mat(self, file_path: Path):
        """Read MAT file and convert to MNE Raw object."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")

        mat_data = loadmat(str(file_path), squeeze_me=False, struct_as_record=False)

        # Try common MAT file structures
        # Structure may vary, try to find data array
        data = None
        sfreq = None
        ch_names = None

        # Common keys to check
        possible_keys = ['data', 'EEG', 'eeg', 'signal', 'channels', 'samples']
        for key in possible_keys:
            if key in mat_data:
                if isinstance(mat_data[key], np.ndarray):
                    data = mat_data[key]
                    break

        if data is None:
            # Try to find largest array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if data is None or value.size > data.size:
                        data = value

        if data is None:
            raise ValueError(f"Could not find data array in MAT file: {file_path}")

        # Reshape if needed: should be (n_channels, n_samples)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T

        # Try to get sampling rate
        if 'sfreq' in mat_data:
            sfreq = float(mat_data['sfreq'].item() if hasattr(mat_data['sfreq'], 'item') else mat_data['sfreq'])
        elif 'fs' in mat_data:
            sfreq = float(mat_data['fs'].item() if hasattr(mat_data['fs'], 'item') else mat_data['fs'])
        else:
            # Default or try to infer
            sfreq = 1000.0  # Common default for EEG

        # Get channel names
        if 'ch_names' in mat_data:
            ch_names = [str(ch) for ch in mat_data['ch_names'].flatten()]
        elif 'channels' in mat_data:
            ch_names = [str(ch) for ch in mat_data['channels'].flatten()]
        else:
            ch_names = [f'EEG{i+1:02d}' for i in range(data.shape[0])]

        # Auto-detect unit and convert to Volts
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        if detected_unit != "V":
            print(f"  Detected unit: {detected_unit}, converted to V")

        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names[:data.shape[0]],
            sfreq=sfreq,
            ch_types=['eeg'] * data.shape[0]
        )

        raw = mne.io.RawArray(data_volts, info, verbose=False)
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")

        return raw

    def _read_raw(self, file_path: Path):
        """Read raw EEG file based on format."""
        file_format = self._detect_file_format()

        if file_format == "edf":
            return self._read_raw_edf(file_path)
        elif file_format == "mat":
            return self._read_raw_mat(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Notch filter
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
            subject_id: Subject identifier (1-25)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building SHU-MI dataset")

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        all_trials = []
        ch_names = None
        trial_counter = 0

        # Process each session
        for session_id, file_info in sorted(files.items()):
            data_file = file_info['data']
            event_file = file_info['events']

            print(f"Reading {data_file}")

            # Read raw data
            raw = self._read_raw(data_file)
            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names
                if self.orig_sfreq is None:
                    self.orig_sfreq = raw.info['sfreq']

            # Read events
            events = self._read_events_tsv(event_file)
            data = raw.get_data()

            # Process each trial
            for event in events:
                onset_sample = int(event['onset'] * self.target_sfreq)
                duration_samples = int(event['duration'] * self.target_sfreq)
                end_sample = onset_sample + duration_samples

                if end_sample <= data.shape[1]:
                    # Data from MNE is in Volts (V), convert to microvolts (µV)
                    trial_data_v = data[:, onset_sample:end_sample]
                    trial_data_uv = trial_data_v * 1e6  # Convert V to µV

                    # Validate trial amplitude
                    self.total_trials += 1
                    if not self._validate_trial(trial_data_uv):
                        max_amp = np.abs(trial_data_uv).max()
                        print(f"  Skipping trial {trial_counter}: amplitude {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                        self.rejected_trials += 1
                        continue

                    self.valid_trials += 1
                    all_trials.append({
                        'data': trial_data_uv,  # Store in µV
                        'label': event['label'],
                        'session_id': session_id,
                        'trial_id': trial_counter,
                        'onset_time': event['onset'],
                    })
                    trial_counter += 1

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="SHU_MI_2class",
            task_type="motor_imaginary",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SHU_MI_INFO.num_labels,
            category_list=SHU_MI_INFO.category_list,
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
            for trial in all_trials:
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # Single segment per trial (window_sec = trial duration)
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
        info = {
            "dataset": {
                "name": SHU_MI_INFO.dataset_name,
                "description": "SHU Motor Imagery Dataset (BIDS format)",
                "task_type": str(SHU_MI_INFO.task_type.value),
                "downstream_task": str(SHU_MI_INFO.downstream_task_type.value),
                "num_labels": SHU_MI_INFO.num_labels,
                "category_list": SHU_MI_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq if self.orig_sfreq else "unknown",
                "channels": "detected from data",
                "montage": SHU_MI_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds004504",
                "num_subjects": 25,
                "num_sessions_per_subject": 5,
                "num_trials_per_session": 100,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "file_format": self.file_format,
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


def build_shu_mi(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build SHU-MI dataset.

    Args:
        raw_data_dir: Directory containing BIDS dataset (with sub-* folders)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for SHUMIBuilder

    Returns:
        List of output file paths
    """
    builder = SHUMIBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SHU-MI HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing BIDS dataset (with sub-* folders)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="auto", choices=["auto", "edf", "mat"], help="File format")
    args = parser.parse_args()

    build_shu_mi(args.raw_data_dir, args.output_dir, args.subjects, file_format=args.format)

