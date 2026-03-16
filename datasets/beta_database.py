"""
BETA Database Dataset Builder.

BETA (BEnchmark database Towards BCI Application) Database - SSVEP BCI Dataset
- 70 subjects
- 40 targets (SSVEP frequencies)
- 64 channels
- 4 sessions/blocks per subject
- 250 Hz sampling rate
- Task: SSVEP-based Brain-Computer Interface
- Reference: https://bci.med.tsinghua.edu.cn/
- Paper: BETA: A Large Benchmark Database Toward SSVEP-BCI Application (Frontiers in Neuroscience, 2020)
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np

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


# Dataset Configuration
BETA_DATABASE_INFO = DatasetInfo(
    dataset_name="BETA_Database",
    task_type=DatasetTaskType.COGNITIVE,  # SSVEP is a cognitive task
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=40,  # 40 targets
    category_list=[f"target_{i+1}" for i in range(40)],  # target_1 to target_40
    sampling_rate=250.0,  # 250 Hz
    montage="10_20",
    channels=[
        # 64 channels - standard 10-20 system
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
        "CZ", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
        "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5",
        "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
    ],
)

# Channels to remove (reference channels, if any)
REMOVE_CHANNELS = []  # Check dataset documentation for reference channels

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for processing.
    
    MNE internally uses Volts (V) as the unit.
    """
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    if max_amp > 1e-2:  # > 0.01, likely microvolts (µV)
        return data / 1e6, 'µV'
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts (mV)
        return data / 1e3, 'mV'
    else:  # likely already Volts (V)
        return data, 'V'


class BETADatabaseBuilder:
    """Builder for BETA Database Dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz for Asia/Europe
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        # Ensure output_dir is absolute path
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        self.output_dir = output_path / "BETA_Database"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = BETA_DATABASE_INFO.sampling_rate  # 250 Hz
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # Validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[str]:
        """
        Get list of subject IDs.
        
        Dataset structure: data/S1.mat, data/S2.mat, ..., data/S70.mat
        """
        data_dir = self.raw_data_dir / "data"
        if not data_dir.exists():
            data_dir = self.raw_data_dir
        
        mat_files = sorted(data_dir.glob("S*.mat"))
        subject_ids = []
        for mat_file in mat_files:
            # Extract subject ID: S1.mat -> "S1"
            subject_id = mat_file.stem
            subject_ids.append(subject_id)
        
        return sorted(subject_ids, key=lambda x: (len(x), x))  # Sort: S1-S9, then S10-S70

    def _find_files(self, subject_id: str) -> list[Path]:
        """
        Find files for a subject.
        
        Dataset structure: data/S1.mat, data/S2.mat, ...
        """
        data_dir = self.raw_data_dir / "data"
        if not data_dir.exists():
            data_dir = self.raw_data_dir
        
        mat_file = data_dir / f"{subject_id}.mat"
        if mat_file.exists():
            return [mat_file]
        return []

    def _read_mat_file(self, file_path: Path):
        """
        Read MATLAB .mat file and extract EEG data.
        
        Data structure:
        - data['data']: structured array with ('EEG', 'suppl_info')
        - EEG shape: (64, 750, 4, 40) = (channels, timepoints, sessions, targets)
        - suppl_info contains: sub, age, gender, chan, freqs, phases, bci_quotient, srate
        
        Returns:
            tuple: (eeg_data, suppl_info, channel_names)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required to read .mat files")
        
        mat_data = loadmat(str(file_path), squeeze_me=False)
        
        if 'data' not in mat_data:
            raise ValueError(f"'data' not found in {file_path}")
        
        data_struct = mat_data['data'][0, 0]
        eeg_data = data_struct['EEG']  # shape: (64, 750, 4, 40)
        suppl_info = data_struct['suppl_info'][0, 0]
        
        # Extract channel names from suppl_info
        chan_info = suppl_info['chan']  # shape: (64, 4)
        # Channel names are in the 4th column (index 3)
        channel_names = []
        for i in range(chan_info.shape[0]):
            chan_name = chan_info[i, 3][0] if isinstance(chan_info[i, 3], np.ndarray) else str(chan_info[i, 3])
            channel_names.append(chan_name.strip())
        
        # Extract frequencies and phases
        freqs = suppl_info['freqs'][0, 0] if suppl_info['freqs'].ndim > 0 else suppl_info['freqs']
        phases = suppl_info['phases'][0, 0] if suppl_info['phases'].ndim > 0 else suppl_info['phases']
        srate = suppl_info['srate'][0, 0] if suppl_info['srate'].ndim > 0 else suppl_info['srate']
        
        return eeg_data, suppl_info, channel_names, freqs, phases, srate

    def _extract_trials_from_mat(self, eeg_data: np.ndarray, session_id: int):
        """
        Extract trials from MATLAB data.
        
        Data format: EEG shape (64 channels, 750 timepoints, 4 sessions, 40 targets)
        Each (session, target) combination is a trial
        
        Args:
            eeg_data: EEG data array, shape (n_channels, n_timepoints, n_sessions, n_targets)
            session_id: Session ID (0-3)
        
        Returns:
            List of trial dictionaries
        """
        trials = []
        n_channels, n_timepoints, n_sessions, n_targets = eeg_data.shape
        
        # Extract trials for this session
        for target_idx in range(n_targets):
            trial_data = eeg_data[:, :, session_id, target_idx]  # (n_channels, n_timepoints)
            
            # Convert to Volts (assuming data is in microvolts)
            trial_data_volts, _ = detect_unit_and_convert_to_volts(trial_data)
            
            trials.append({
                'data': trial_data_volts,
                'label': target_idx,  # Label is target index (0-39)
                'trial_id': target_idx,
                'session_id': session_id + 1,  # Session ID: 1-4
                'onset_time': 0.0,
            })
        
        return trials

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def _preprocess_data(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Apply preprocessing to data array.
        
        Note: This is a simplified version. For full preprocessing with filtering,
        MNE is recommended.
        
        Args:
            data: Data array, shape (n_channels, n_samples)
            sfreq: Sampling frequency
        
        Returns:
            Preprocessed data array
        """
        # Resample if needed
        if sfreq != self.target_sfreq:
            from scipy import signal
            n_samples_new = int(data.shape[1] * self.target_sfreq / sfreq)
            data_resampled = np.zeros((data.shape[0], n_samples_new))
            for ch_idx in range(data.shape[0]):
                data_resampled[ch_idx, :] = signal.resample(data[ch_idx, :], n_samples_new)
            data = data_resampled
        
        return data

    def _save_dataset_info(self, stats: dict, channel_names: list = None, freqs: np.ndarray = None, phases: np.ndarray = None) -> None:
        """Save dataset info and processing parameters to JSON."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        channels_to_save = channel_names if channel_names else BETA_DATABASE_INFO.channels
        
        info = {
            "dataset": {
                "name": BETA_DATABASE_INFO.dataset_name,
                "description": "BETA Database - A Large Benchmark Database Toward SSVEP-BCI Application",
                "task_type": str(BETA_DATABASE_INFO.task_type.value),
                "downstream_task": str(BETA_DATABASE_INFO.downstream_task_type.value),
                "num_labels": BETA_DATABASE_INFO.num_labels,
                "category_list": BETA_DATABASE_INFO.category_list,
                "original_sampling_rate": BETA_DATABASE_INFO.sampling_rate,
                "channels": channels_to_save,
                "channel_count": len(channels_to_save),
                "montage": BETA_DATABASE_INFO.montage,
                "source_url": "https://bci.med.tsinghua.edu.cn/",
                "reference": "BETA: A Large Benchmark Database Toward SSVEP-BCI Application (Frontiers in Neuroscience, 2020)",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "removed_channels": REMOVE_CHANNELS,
                "max_amplitude_uv": self.max_amplitude_uv,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }
        
        # Add frequency and phase information if available
        if freqs is not None:
            # Handle nested arrays
            if isinstance(freqs, np.ndarray):
                if freqs.ndim > 1:
                    freqs = freqs.flatten()
                info["dataset"]["target_frequencies"] = freqs.tolist()
            else:
                info["dataset"]["target_frequencies"] = freqs
        if phases is not None:
            # Handle nested arrays
            if isinstance(phases, np.ndarray):
                if phases.ndim > 1:
                    phases = phases.flatten()
                info["dataset"]["target_phases"] = phases.tolist()
            else:
                info["dataset"]["target_phases"] = phases

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"Saved dataset info to {json_path}")

    def build_subject(self, subject_id: str) -> str:
        """Build HDF5 file for a single subject."""
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = None
        freqs = None
        phases = None

        # Process the .mat file
        file_path = files[0]
        print(f"Reading {file_path}")
        
        try:
            # Read MATLAB file
            eeg_data, suppl_info, channel_names, target_freqs, target_phases, srate = self._read_mat_file(file_path)
            
            print(f"  Data shape: {eeg_data.shape}")
            print(f"  Sampling rate: {srate} Hz")
            print(f"  Channels: {len(channel_names)} channels")
            print(f"  Sessions: {eeg_data.shape[2]}")
            print(f"  Targets: {eeg_data.shape[3]}")
            
            if ch_names is None:
                ch_names = channel_names
                freqs = target_freqs
                phases = target_phases
            
            # Extract trials from all sessions
            n_sessions = eeg_data.shape[2]
            for session_idx in range(n_sessions):
                trials = self._extract_trials_from_mat(eeg_data, session_idx)
                print(f"  Extracted {len(trials)} trials from session {session_idx + 1}")
                all_trials.extend(trials)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=BETA_DATABASE_INFO.dataset_name,
            task_type=BETA_DATABASE_INFO.task_type.value,
            downstream_task_type=BETA_DATABASE_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=BETA_DATABASE_INFO.num_labels,
            category_list=BETA_DATABASE_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=BETA_DATABASE_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            trial_counter = 0
            for trial in all_trials:
                trial_data = trial["data"]  # Already in Volts, shape: (n_channels, n_samples)
                
                # Resample if needed
                if self.orig_sfreq != self.target_sfreq:
                    trial_data = self._preprocess_data(trial_data, self.orig_sfreq)
                
                # Convert to µV for validation
                trial_data_uv = trial_data * 1e6
                
                # Validate trial amplitude
                self.total_trials += 1
                if not self._validate_trial(trial_data_uv):
                    self.rejected_trials += 1
                    print(f"  Skipping trial {trial['trial_id']} (target {trial['label']+1}, session {trial['session_id']}): amplitude {np.abs(trial_data_uv).max():.1f} µV > {self.max_amplitude_uv} µV")
                    continue
                
                self.valid_trials += 1

                trial_attrs = TrialAttrs(
                    trial_id=trial_counter,
                    session_id=trial.get("session_id", 0),
                )
                trial_name = writer.add_trial(trial_attrs)
                trial_counter += 1

                # Segment into windows
                n_samples = trial_data.shape[1]
                for i_slice, start in enumerate(
                    range(0, n_samples - self.window_samples + 1, self.stride_samples)
                ):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]
                    
                    # Convert from V to µV for export
                    slice_data_uv = slice_data * 1e6

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=trial.get("onset_time", 0.0) + start / self.target_sfreq,
                        end_time=trial.get("onset_time", 0.0) + end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([trial["label"]]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data_uv)

        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[str] | None = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_trials = 0
        all_valid_trials = 0
        all_rejected_trials = 0
        
        # Store channel names and frequency info from first subject
        first_ch_names = None
        first_freqs = None
        first_phases = None

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
                
                # Store metadata from first subject
                if first_ch_names is None:
                    try:
                        files = self._find_files(subject_id)
                        if files:
                            _, _, ch_names, freqs, phases, _ = self._read_mat_file(files[0])
                            first_ch_names = ch_names
                            first_freqs = freqs
                            first_phases = phases
                    except Exception:
                        pass
                        
            except Exception as exc:
                print(f"Error processing subject {subject_id}: {exc}")
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

        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_trials": all_total_trials,
            "valid_trials": all_valid_trials,
            "rejected_trials": all_rejected_trials,
            "rejection_rate": f"{all_rejected_trials / all_total_trials * 100:.1f}%" if all_total_trials > 0 else "0%",
        }
        self._save_dataset_info(stats, first_ch_names, first_freqs, first_phases)

        return output_paths


def build_beta_database(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] | None = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build BETA Database dataset.
    
    Args:
        raw_data_dir: Directory containing raw files (should have data/ subdirectory)
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all, e.g., ['S1', 'S2'])
        **kwargs: Additional arguments for BETADatabaseBuilder
    
    Returns:
        List of output file paths
    """
    builder = BETADatabaseBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build BETA Database HDF5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m benchmark_dataloader.datasets.beta_database /path/to/BETA/Database --output_dir ./hdf5
  python -m benchmark_dataloader.datasets.beta_database /path/to/BETA/Database --subjects S1 S2 S3
        """
    )
    parser.add_argument("raw_data_dir", help="Directory containing raw files (with data/ subdirectory)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=str, help="Subject IDs to process (e.g., S1 S2)")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling frequency")
    parser.add_argument("--window_sec", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Stride length in seconds")
    parser.add_argument("--filter_low", type=float, default=0.1, help="Low cutoff frequency")
    parser.add_argument("--filter_high", type=float, default=75.0, help="High cutoff frequency")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency")
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV, help="Max amplitude threshold (µV)")
    args = parser.parse_args()

    build_beta_database(
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
