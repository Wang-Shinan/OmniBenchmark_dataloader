"""
SEED_FRA Dataset Builder.

SEED_FRA Dataset - French Emotion Recognition Dataset
- Multiple subjects
- 3 sessions per subject
- 15 trials per session (emotion recognition)
- 3 emotion classes: Negative, Neutral, Positive
- 62 EEG channels (10-10 system)
- 1000 Hz sampling rate (original)
- https://bcmi.sjtu.edu.cn/~seed/index.html

"""

import os
import re
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import mne
    from scipy.io import loadmat
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

# SEED standard emotion labels (15 trials per session)
# -1: Negative, 0: Neutral, 1: Positive
SEED_LABELS = [1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}  # Map to 0, 1, 2 for classification

# Standard 62-channel 10-10 system (excluding CB1, CB2 reference channels)
SEED_FRA_CHANNELS = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# Channels to remove (reference electrodes)
SEED_FRA_REMOVE_CHANNELS = ['CB1', 'CB2']

SEED_FRA_INFO = DatasetInfo(
    dataset_name="SEED_FRA",
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=3,
    category_list=["Negative", "Neutral", "Positive"],
    sampling_rate=1000.0,
    montage="10_10",
    channels=[ch for ch in SEED_FRA_CHANNELS if ch not in SEED_FRA_REMOVE_CHANNELS],
)

# Default amplitude threshold (µV)
DEFAULT_MAX_AMPLITUDE_UV = 500.0


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


class SEEDFRABuilder:
    """Builder for SEED_FRA dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        use_blind_slicing: bool = True,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "SEED_FRA"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.use_blind_slicing = use_blind_slicing

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

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        info = {
            "dataset": {
                "name": SEED_FRA_INFO.dataset_name,
                "description": "SEED_FRA Dataset - French Emotion Recognition EEG",
                "task_type": str(SEED_FRA_INFO.task_type.value),
                "downstream_task": str(SEED_FRA_INFO.downstream_task_type.value),
                "num_labels": SEED_FRA_INFO.num_labels,
                "category_list": SEED_FRA_INFO.category_list,
                "original_sampling_rate": SEED_FRA_INFO.sampling_rate,
                "channels": SEED_FRA_INFO.channels,
                "montage": SEED_FRA_INFO.montage,
                "source_url": "https://bcmi.sjtu.edu.cn/~seed/index.html",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "removed_channels": SEED_FRA_REMOVE_CHANNELS,
                "max_amplitude_uv": self.max_amplitude_uv,
                "use_blind_slicing": self.use_blind_slicing,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs from directory structure."""
        # Search for files in French/01-EEG-raw or 01-EEG-raw
        search_path = self.raw_data_dir
        if (search_path / "French" / "01-EEG-raw").exists():
            search_path = search_path / "French" / "01-EEG-raw"
        elif (search_path / "01-EEG-raw").exists():
            search_path = search_path / "01-EEG-raw"

        # Find all .cnt or .mat files
        cnt_files = list(search_path.glob("*.cnt"))
        mat_files = list(search_path.glob("*.mat"))
        
        subject_ids = set()
        for f in cnt_files + mat_files:
            # Extract subject ID from filename (e.g., "1_1.cnt" -> subject_id=1)
            match = re.match(r'(\d+)_\d+', f.stem)
            if match:
                subject_ids.add(int(match.group(1)))
        
        return sorted(list(subject_ids))

    def _find_files(self, subject_id: int) -> dict[int, Path]:
        """Find all session files for a subject."""
        search_path = self.raw_data_dir
        if (search_path / "French" / "01-EEG-raw").exists():
            search_path = search_path / "French" / "01-EEG-raw"
        elif (search_path / "01-EEG-raw").exists():
            search_path = search_path / "01-EEG-raw"

        files = {}
        # Look for .mat files first (converted format), then .cnt files
        for session in range(1, 4):
            mat_file = search_path / f"{subject_id}_{session}.mat"
            cnt_file = search_path / f"{subject_id}_{session}.cnt"
            
            if mat_file.exists():
                files[session] = mat_file
            elif cnt_file.exists():
                files[session] = cnt_file
        
        return files

    def _read_mat(self, file_path: Path):
        """Read .mat file and convert to MNE Raw object."""
        mat = loadmat(str(file_path))
        
        # Find data key (usually 'data' or 'eeg')
        data_key = None
        for key in ['data', 'eeg', 'EEG']:
            if key in mat and isinstance(mat[key], np.ndarray):
                data_key = key
                break
        
        if data_key is None:
            # Try to find the largest array
            for key, value in mat.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                    if value.shape[0] > 10:  # Likely EEG data
                        data_key = key
                        break
        
        if data_key is None:
            raise ValueError(f"Could not find EEG data in {file_path}")
        
        data = mat[data_key]  # shape: (n_channels, n_samples) or (n_samples, n_channels)
        
        # Ensure shape is (n_channels, n_samples)
        if data.shape[0] < data.shape[1]:
            data = data.T
        
        # Auto-detect unit and convert to Volts
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        print(f"  Detected unit: {detected_unit}, max amplitude: {np.abs(data).max():.2e}")

        # Create channel names (use standard SEED channels if available)
        n_channels = data_volts.shape[0]
        if n_channels >= 62:
            ch_names = SEED_FRA_CHANNELS[:n_channels]
        else:
            ch_names = [f'CH{i+1}' for i in range(n_channels)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.orig_sfreq,
            ch_types=['eeg'] * len(ch_names)
        )
        
        raw = mne.io.RawArray(data_volts, info)
        return raw

    def _read_cnt(self, file_path: Path):
        """Read .cnt file and convert to MNE Raw object."""
        try:
            # Try Neuroscan format first
            raw = mne.io.read_raw_cnt(str(file_path), preload=True)
        except Exception as e1:
            try:
                # Try ANT Neuro format (if available)
                raw = mne.io.read_raw_ant(str(file_path), preload=True)
            except Exception as e2:
                raise ValueError(f"Could not read CNT file: {e1}, {e2}")
        
        # Auto-detect unit and convert to Volts
        max_amp = np.abs(raw._data).max()
        if max_amp > 1e-3:  # > 0.001, likely microvolts
            raw._data = raw._data / 1e6
            detected_unit = "µV"
        else:
            detected_unit = "V"
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_amp:.2e}")
        
        return raw

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop reference channels
        channels_to_drop = [ch for ch in SEED_FRA_REMOVE_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
            print(f"  Dropped reference channels: {channels_to_drop}")

        # Keep only first 62 channels if more are present
        if len(raw.ch_names) > 62:
            raw.pick(raw.ch_names[:62])
            print(f"  Kept first 62 channels")

        # Notch filter
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch)

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high)

        # Resample if needed
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq)

        return raw

    def _extract_trials_blind_slicing(self, data: np.ndarray, sfreq: float) -> list[dict]:
        """Extract trials using blind slicing (15 trials per session)."""
        trials = []
        num_trials = 15
        total_samples = data.shape[1]
        block_size = total_samples // num_trials
        capture_duration = 180  # seconds
        capture_samples = int(capture_duration * sfreq)
        
        if capture_samples > block_size:
            capture_samples = int(block_size * 0.9)
        
        for i in range(num_trials):
            block_start = i * block_size
            block_center = block_start + (block_size // 2)
            start_idx = block_center - (capture_samples // 2)
            end_idx = block_center + (capture_samples // 2)
            
            if start_idx < 0:
                start_idx = 0
            if end_idx > total_samples:
                end_idx = total_samples
            
            raw_label = SEED_LABELS[i]
            final_label = LABEL_MAP.get(raw_label, 1)
            
            trials.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'label': final_label,
                'trial_id': i,
            })
        
        return trials

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (e.g., 1, 2, 3, ...)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE and scipy are required for building SEED_FRA dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        print(f"Processing subject {subject_id}, found {len(files)} session(s)")

        all_trials = []
        ch_names = None
        trial_counter = 0

        # Process each session
        for session_id, file_path in sorted(files.items()):
            print(f"  Reading Session {session_id}: {file_path.name}")

            try:
                # Read file based on extension
                if file_path.suffix == '.mat':
                    raw = self._read_mat(file_path)
                elif file_path.suffix == '.cnt':
                    raw = self._read_cnt(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")

                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = raw.ch_names

                # Get data (in Volts)
                data = raw.get_data()  # shape: (n_channels, n_samples)

                # Extract trials
                if self.use_blind_slicing:
                    trials = self._extract_trials_blind_slicing(data, raw.info['sfreq'])
                else:
                    # If annotations are available, use them
                    trials = []
                    if len(raw.annotations) > 0:
                        for ann in raw.annotations:
                            start_idx = int(ann['onset'] * raw.info['sfreq'])
                            duration_samples = int(ann['duration'] * raw.info['sfreq'])
                            end_idx = start_idx + duration_samples
                            
                            # Get label from annotation description
                            label = 1  # Default to neutral
                            if 'negative' in ann['description'].lower() or '-1' in ann['description']:
                                label = 0
                            elif 'positive' in ann['description'].lower() or '1' in ann['description']:
                                label = 2
                            
                            trials.append({
                                'start_idx': start_idx,
                                'end_idx': end_idx,
                                'label': label,
                                'trial_id': len(trials),
                            })
                    else:
                        # Fallback to blind slicing
                        trials = self._extract_trials_blind_slicing(data, raw.info['sfreq'])

                # Process each trial
                for trial in trials:
                    trial_data = data[:, trial['start_idx']:trial['end_idx']]
                    
                    # Convert to µV for validation
                    trial_data_uv = trial_data * 1e6
                    self.total_trials += 1
                    
                    # Validate trial amplitude
                    if not self._validate_trial(trial_data_uv):
                        self.rejected_trials += 1
                        print(f"  Skipping trial {trial['trial_id']}: amplitude {np.abs(trial_data_uv).max():.1f} µV > {self.max_amplitude_uv} µV")
                        continue
                    
                    self.valid_trials += 1
                    
                    all_trials.append({
                        'data': trial_data_uv,
                        'label': trial['label'],
                        'trial_id': trial_counter,
                        'session_id': session_id,
                    })
                    trial_counter += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

        if not all_trials:
            raise ValueError(f"No valid trials extracted for subject {subject_id}")

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SEED_FRA_INFO.dataset_name,
            task_type=SEED_FRA_INFO.task_type.value,
            downstream_task_type=SEED_FRA_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SEED_FRA_INFO.num_labels,
            category_list=SEED_FRA_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=SEED_FRA_INFO.montage,
        )

        # Create output file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # Segment into windows
                data = trial['data']  # shape = (n_chs, n_times) in µV
                n_chs, total_samples = data.shape

                start_sample = 0
                seg_id = 0

                while start_sample + self.window_samples <= total_samples:
                    end_sample = start_sample + self.window_samples
                    seg_data = data[:, start_sample:end_sample]

                    seg_start_time = start_sample / self.target_sfreq
                    seg_end_time = end_sample / self.target_sfreq

                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=seg_start_time,
                        end_time=seg_end_time,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)

                    start_sample += self.stride_samples
                    seg_id += 1

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


def build_seed_fra(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build SEED_FRA dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for SEEDFRABuilder

    Returns:
        List of output file paths
    """
    builder = SEEDFRABuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SEED_FRA HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate")
    parser.add_argument("--window_sec", type=float, default=4.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=4.0, help="Stride length in seconds")
    parser.add_argument("--filter_notch", type=float, default=50.0, help="Notch filter frequency")
    args = parser.parse_args()

    build_seed_fra(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_notch=args.filter_notch,
    )
