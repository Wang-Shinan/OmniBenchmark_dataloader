"""
160 Targets SSVEP BCI Dataset Builder.

160 Targets SSVEP BCI Dataset - Steady-State Visual Evoked Potential Brain-Computer Interface
- ## TODO: Update with actual dataset information
- N subjects
- M sessions per subject (if applicable)
- K trials per session
- 160 classes (targets)
- https://bci.med.tsinghua.edu.cn/
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
    import mne
    from mne import create_info
    from mne.io import RawArray
    HAS_MNE = True
except ImportError:
    try:
        import mne
        create_info = mne.create_info
        RawArray = mne.io.RawArray
        HAS_MNE = True
    except (ImportError, AttributeError):
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
SSVEP_160TARGETS_INFO = DatasetInfo(
    dataset_name="SSVEP_160Targets",
    task_type=DatasetTaskType.COGNITIVE,  # SSVEP is a cognitive task
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=160,  # 160 targets
    category_list=[f"target_{i+1}" for i in range(160)],  # target_1 to target_160
    sampling_rate=250.0,  # Downsampled from original (1035 samples suggests ~250Hz for ~4s trials)
    montage="10_20",  # ## TODO: Update if different (10_10, custom, etc.)
    channels=[
        # 9 channels for SSVEP dataset
       "Pz", "PO5" , "PO3" , "POz" , "PO4" , "PO6" , "O1" , "Oz", "O2"
    ],
)

# Label mapping: target ID -> class index (0-159)
# ## TODO: Update if label mapping is different
LABEL_MAPPING = {i+1: i for i in range(160)}  # target_1 -> 0, target_2 -> 1, ..., target_160 -> 159

# Channels to remove (reference channels, trigger channels, etc.)
# ## TODO: List channels to remove (e.g., ['CB1', 'CB2', 'M1', 'M2', 'Trigger'])
REMOVE_CHANNELS = []

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 600.0


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.
    
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


class SSVEP160TargetsBuilder:
    """Builder for 160 Targets SSVEP BCI Dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz for Asia/Europe, 60Hz for Americas
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        # Ensure output_dir is absolute path
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        self.output_dir = output_path / "SSVEP_160Targets"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = SSVEP_160TARGETS_INFO.sampling_rate  # Original sampling rate
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
        
        Dataset structure: data/S1/, data/S2/, ..., data/S8/, data/SS1/, data/SS2/, ..., data/SS12/
        """
        data_dir = self.raw_data_dir / "data"
        if not data_dir.exists():
            data_dir = self.raw_data_dir
        
        subject_dirs = sorted(data_dir.glob("S*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            # Extract subject ID: S1 -> "S1", SS1 -> "SS1"
            subject_ids.append(sub_dir.name)
        
        return sorted(subject_ids, key=lambda x: (len(x), x))  # Sort: S1-S8, then SS1-SS12

    def _find_files(self, subject_id: int) -> list[Path]:
        """
        Find all files for a subject.
        
        ## TODO: Update based on actual file naming convention
        """
        data_dir = self.raw_data_dir / "data"
        if not data_dir.exists():
            data_dir = self.raw_data_dir
        
        sub_dir = data_dir / subject_id
        if not sub_dir.exists():
            return []
        
        # Find all .mat files in subject directory
        files = sorted(sub_dir.glob("*.mat"))
        return files

    def _read_mat_file(self, file_path: Path):
        """
        Read MATLAB .mat file and extract EEG data.
        
        Data format: EEG_downsample shape (160 targets, 9 channels, 1035 samples)
        Returns: eeg_data array and metadata
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required to read .mat files")
        
        mat_data = loadmat(str(file_path), squeeze_me=False)
        
        if 'EEG_downsample' not in mat_data:
            raise ValueError(f"EEG_downsample not found in {file_path}")
        
        eeg_data = mat_data['EEG_downsample']  # shape: (160, 9, 1035)
        # Remove singleton dimensions
        while isinstance(eeg_data, np.ndarray) and eeg_data.ndim > 3 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]
        
        if not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 3:
            raise ValueError(f"Unexpected EEG_downsample shape: {eeg_data.shape}, expected (targets, channels, samples)")
        
        n_targets, n_channels, n_samples = eeg_data.shape
        
        return eeg_data, n_targets, n_channels, n_samples

    def _extract_trials_from_mat(self, eeg_data: np.ndarray, file_session_id: int = 0):
        """
        Extract trials from MATLAB data.
        
        Data format: EEG_downsample shape (160 targets, 9 channels, 1035 samples)
        Each target is a separate trial with label = target_index (0-159)
        
        Args:
            eeg_data: EEG data array, shape (n_targets, n_channels, n_samples)
            file_session_id: Session ID for this file
        
        Returns:
            List of trial dictionaries
        """
        trials = []
        n_targets, n_channels, n_samples = eeg_data.shape
        
        # Extract each target as a separate trial
        for target_idx in range(n_targets):
            target_data = eeg_data[target_idx, :, :]  # (n_channels, n_samples)
            
            # Convert to Volts (assuming data is in microvolts)
            target_data_volts, _ = detect_unit_and_convert_to_volts(target_data)
            
            trials.append({
                'data': target_data_volts,
                'label': target_idx,  # Label is target index (0-159)
                'trial_id': target_idx,
                'session_id': file_session_id,
                'onset_time': 0.0,
            })
        
        return trials

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

    def _preprocess_data(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Apply preprocessing to data array.
        
        Note: This is a simplified version without MNE. For full preprocessing,
        MNE is recommended.
        
        Args:
            data: Data array, shape (n_channels, n_samples)
            sfreq: Sampling frequency
        
        Returns:
            Preprocessed data array
        """
        # For now, just resample if needed
        # Full filtering would require MNE or scipy.signal
        if sfreq != self.target_sfreq:
            from scipy import signal
            n_samples_new = int(data.shape[1] * self.target_sfreq / sfreq)
            data_resampled = np.zeros((data.shape[0], n_samples_new))
            for ch_idx in range(data.shape[0]):
                data_resampled[ch_idx, :] = signal.resample(data[ch_idx, :], n_samples_new)
            data = data_resampled
        
        return data

    def _save_dataset_info(self, stats: dict) -> None:
        """Save dataset info and processing parameters to JSON."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        info = {
            "dataset": {
                "name": SSVEP_160TARGETS_INFO.dataset_name,
                "description": "160 Targets SSVEP BCI Dataset - Steady-State Visual Evoked Potential Brain-Computer Interface",
                "task_type": str(SSVEP_160TARGETS_INFO.task_type.value),
                "downstream_task": str(SSVEP_160TARGETS_INFO.downstream_task_type.value),
                "num_labels": SSVEP_160TARGETS_INFO.num_labels,
                "category_list": SSVEP_160TARGETS_INFO.category_list,
                "original_sampling_rate": SSVEP_160TARGETS_INFO.sampling_rate,
                "channels": SSVEP_160TARGETS_INFO.channels,
                "channel_count": len(SSVEP_160TARGETS_INFO.channels),
                "montage": SSVEP_160TARGETS_INFO.montage,
                "source_url": "https://bci.med.tsinghua.edu.cn/",
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
        n_channels = None

        for session_id, file_path in enumerate(files, 1):
            print(f"Reading {file_path}")
            
            try:
                # Read MATLAB file
                eeg_data, n_targets, n_ch, n_samples = self._read_mat_file(file_path)
                
                if ch_names is None:
                    n_channels = n_ch
                    ch_names = SSVEP_160TARGETS_INFO.channels[:n_channels] if len(SSVEP_160TARGETS_INFO.channels) >= n_channels else [f"CH{i+1}" for i in range(n_channels)]
                    print(f"  Data shape: ({n_targets}, {n_channels}, {n_samples})")
                    print(f"  Channels: {ch_names}")
                
                # Extract trials
                trials = self._extract_trials_from_mat(eeg_data, file_session_id=session_id)
                print(f"  Extracted {len(trials)} trials from {n_targets} targets")
                
                all_trials.extend(trials)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SSVEP_160TARGETS_INFO.dataset_name,
            task_type=SSVEP_160TARGETS_INFO.task_type.value,
            downstream_task_type=SSVEP_160TARGETS_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SSVEP_160TARGETS_INFO.num_labels,
            category_list=SSVEP_160TARGETS_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=SSVEP_160TARGETS_INFO.montage,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use subject_id as-is (S1, S2, SS1, etc.)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            trial_counter = 0
            for trial in all_trials:
                trial_data = trial["data"]  # Already in Volts, shape: (n_channels, n_samples)
                
                # Convert to µV for validation
                trial_data_uv = trial_data * 1e6
                
                # Validate trial amplitude
                self.total_trials += 1
                if not self._validate_trial(trial_data_uv):
                    self.rejected_trials += 1
                    print(f"  Skipping trial {trial['trial_id']} (target {trial['label']+1}): amplitude {np.abs(trial_data_uv).max():.1f} µV > {self.max_amplitude_uv} µV")
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

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_trials += self.total_trials
                all_valid_trials += self.valid_trials
                all_rejected_trials += self.rejected_trials
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
        self._save_dataset_info(stats)

        return output_paths


def build_ssvep_160targets(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] | None = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build 160 Targets SSVEP BCI dataset.
    
    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for SSVEP160TargetsBuilder
    
    Returns:
        List of output file paths
    """
    builder = SSVEP160TargetsBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build 160 Targets SSVEP BCI HDF5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m benchmark_dataloader.datasets.ssvep_160targets /path/to/raw/data --output_dir ./hdf5
  python -m benchmark_dataloader.datasets.ssvep_160targets /path/to/raw/data --subjects 1 2 3
        """
    )
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=str, help="Subject IDs to process (e.g., S1 S2 SS1)")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling frequency")
    parser.add_argument("--window_sec", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Stride length in seconds")
    parser.add_argument("--filter_low", type=float, default=0.1, help="Low cutoff frequency")
    parser.add_argument("--filter_high", type=float, default=75.0, help="High cutoff frequency")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency")
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV, help="Max amplitude threshold (µV)")
    args = parser.parse_args()

    build_ssvep_160targets(
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
