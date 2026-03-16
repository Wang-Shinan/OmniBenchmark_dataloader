"""
DEAP Dataset Builder.

DEAP Dataset: Emotional Dynamics and Physiological Appraisal.
- 32 subjects (ID: 1-32)
- 40 trials per subject
- 32 channels
"""

import os
import warnings
from pathlib import Path
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

DEAP_INFO = DatasetInfo(
    dataset_name="DEAP",
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=8,
    category_list=[],
    sampling_rate=128.0,
    montage="10_20",
    channels=["Fp1", "AF3", "F7", "F3", "FC1",
     "FC5", "T7", "C3", "CP1", "CP5",
      "P7", "P3", "Pz", "PO3","O1",
      "Oz","O2","PO4","P4","P8",
      "CP6","CP2","C4","T8","FC6",
      "FC2","F4","F8","AF4","Fp2",
      "Fz","Cz"],
)

# Channels to remove (EOG)
REMOVE_CHANNELS = []

class DEAPBuilder:
    """Builder for DEAP dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 250.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        file_format: str = "auto",   # 'auto', 'gdf', or 'mat'
    ):
        """
        Initialize DEAP builder.

        Args:
            raw_data_dir: Directory containing raw files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50Hz for Europe)
            file_format: File format ('auto', 'gdf', or 'mat')
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "DEAP"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 250.0  # Original sampling rate
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.file_format = file_format

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-32)."""
        return list(range(1, 33))

    def _detect_file_format(self) -> str:
        """Auto-detect file format from directory structure."""
        gdf_files = list(self.raw_data_dir.rglob("*.gdf"))
        bdf_files = list(self.raw_data_dir.rglob("*.bdf"))
        mat_files = list(self.raw_data_dir.rglob("*.mat"))
        
        if gdf_files and mat_files and bdf_files:
            print("Warning: Both GDF BDF and MAT files found. Using GDF format.")
            return "gdf"
        elif gdf_files:
            return "gdf"
        elif mat_files:
            return "mat"
        elif bdf_files:
            return "bdf"
        else:
            raise FileNotFoundError("No GDF or MAT files found in directory.")

    def _find_files(self, subject_id: int) -> dict[str, list[Path]]:
        """
        仅返回单个被试的所有 session 文件。
        目录结构：
            <raw_data_dir>/sub-01/ses-01/..._task-CIRE_eeg.edf
        """
        ext = '.edf'
        sub_dir: Path = self.raw_data_dir / f'sub-{subject_id:02d}'

        if not sub_dir.exists():
            raise FileNotFoundError(f'被试目录不存在: {sub_dir}')

        data: dict[str, list[Path]] = {}
        # 只在该被试目录内递归
        for file in sub_dir.rglob(f'*_task-CIRE_eeg{ext}'):
            # 抓出 session 名（ses-xx）
            ses_match = re.search(r'ses-\d+', file.name)
            if not ses_match:
                continue
            ses = ses_match.group()          # 'ses-01'
            data.setdefault(ses, []).append(file)

        # 每个 session 文件按名排序
        for ses in data:
            data[ses] = sorted(data[ses])

        return data

    def _read_raw_gdf(self, file_path: Path):
        """Read raw GDF file."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose=False)
        return raw

    def _read_raw_mat(self, file_path: Path):
        """
        Read raw MAT file and convert to MNE Raw object.
    
        BCIC-2A MAT files structure:
        - 'data': List of runs (typically 9 runs: 3 preparation + 6 experimental)
            Each run dict contains:
        - 'X': EEG data (n_samples, n_channels) - continuous data stream
        - 'y': Labels (n_trials,) - 1-4 for classes, empty for preparation runs
        - 'trial': Trial start samples (n_trials,) - sample indices, empty for prep runs
        - 'fs': Sampling frequency (250 Hz)
        - 'classes': Class names

        Args:
            file_path: Path to MAT file

        Returns:
            mne.io.Raw: MNE Raw object
        """
        mat_data = loadmat(str(file_path),simplify_cells=True)

        if 'data' not in mat_data:
            raise ValueError("MAT file does not contain 'data' key.")

        data_list = mat_data['data']
        if not isinstance(data_list, list) or len(data_list) == 0:
            raise ValueError(f"Invalid data structure in MAT file: {file_path}")

        # Process each run
        all_runs_data = []
        all_runs_labels = []
        all_runs_trials = []
        ch_names = None
        sfreq = self.orig_sfreq

        for run_idx, run_dict in enumerate(data_list):
            if not isinstance(run_dict, dict) or 'X' not in run_dict:
                print(f"  Warning: Run {run_idx+1} missing 'X' key, skipping")
                continue
            
            X = run_dict['X']
            y = run_dict.get('y', np.array([]))
            trial = run_dict.get('trial', np.array([]))
            fs = run_dict.get('fs', self.orig_sfreq)

            # X shape: (n_samples, n_channels)
            if not isinstance(X, np.ndarray) or X.ndim != 2:
                print(f"  Warning: Invalid X shape in run {run_idx+1}, skipping")
                continue

            # Transpose to (n_channels, n_samples) for MNE
            X_T = X.T
            
            # Get sampling frequency
            if isinstance(fs, (int, float)):
                sfreq = float(fs)
            elif isinstance(fs, np.ndarray):
                sfreq = float(fs.item())
            
            # Store channel names from first run
            if ch_names is None:
                n_channels = X_T.shape[0]
                if n_channels == 22:
                    ch_names = BCIC2A_INFO.channels
                elif n_channels == 25:  # 22 EEG + 3 EOG
                    ch_names = BCIC2A_INFO.channels + BCIC2A_REMOVE_CHANNELS
                else:
                    ch_names = [f'EEG{i+1}' for i in range(n_channels)]

            all_runs_data.append(X_T)
        
            # Store labels and trial markers
            if len(y) > 0:
                all_runs_labels.append(y)
            else:
                all_runs_labels.append(None)
            
            if len(trial) > 0:
                all_runs_trials.append(trial)
            else:
                all_runs_trials.append(None)

        if not all_runs_data:
            raise ValueError("No valid data found in MAT file.")

        # Concatenate all runs along time axis
        concatenated_data = np.concatenate(all_runs_data, axis=1)
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * len(ch_names),
        )

        # Convert to volts if needed
        if np.abs(concatenated_data).max() < 1.0:
            raw = mne.io.RawArray(concatenated_data, info, verbose=False)
        else:
            # Assume microvolts, convert to volts
            raw = mne.io.RawArray(concatenated_data / 1e6, info, verbose=False)

        # Add annotations from labels
        current_time = 0.0
        
        for run_idx, (labels, trials) in enumerate(zip(all_runs_labels, all_runs_trials)):
            run_samples = all_runs_data[run_idx].shape[1]

            if labels is not None and len(labels) > 0 and trials is not None and len(trials) > 0:
                if len(labels) != len(trials):
                    print(f"  Warning: Run {run_idx+1} label/trial count mismatch, skipping annotations")
                    current_time += run_samples / sfreq
                    continue

                # Add annotations for each trial
                for label, trial_start_sample in zip(labels, trials):
                    mat_label = int(label)
                    if mat_label in MAT_LABEL_TO_BCIC:
                        bcic_label = MAT_LABEL_TO_BCIC[mat_label]
                        if bcic_label in BCIC2A_LABEL_MAP:
                            onset = current_time + trial_start_sample / sfreq
                            raw.annotations.append(
                                onset=onset,
                                duration=0.0,
                                description=bcic_label
                            )

            # Advance time offset for next run
            current_time += run_samples / sfreq
        
        return raw


    def _read_raw(self, file_path: Path):
        """Read raw file (GDF or MAT)."""
        if self.file_format == "gdf":
            return self._read_raw_gdf(file_path)
        elif self.file_format == "mat":
            return self._read_raw_mat(file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")            
        
    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop EOG channels if present
        channels_to_drop = [ch for ch in BCIC2A_REMOVE_CHANNELS if ch in raw.ch_names]
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

    def _extract_trials(self, raw) -> list[dict]:
        """Extract trials from annotations."""
        trials = []
        anno = raw.annotations
        
        for onset, desc in zip(anno.onset, anno.description):
            if desc in BCIC2A_LABEL_MAP:
                label = BCIC2A_LABEL_MAP[desc]
                trials.append({
                    'onset': onset,
                    'label': label,
                })
        
        return trials

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (1-9)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE and scipy are required for building BCIC-2A dataset")

        files_by_session = self._find_files(subject_id)
        
        # Check if we have any files
        total_files = sum(len(files) for files in files_by_session.values())
        if total_files == 0:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        all_trials = []
        ch_names = None
        trial_counter = 0
        
        # Process sessions: T (training=1) and E (evaluation=2)
        session_mapping = {'T': 1, 'E': 2}
        
        for session_type in ['T', 'E']:
            session_id = session_mapping[session_type]
            files = files_by_session[session_type]
            
            if not files:
                print(f"No {session_type} session files found for subject {subject_id}")
                continue
            
            # Track cumulative time within session
            session_time_offset = 0.0
            
            # Process each file in this session
            for file_idx, file_path in enumerate(files, 1):
                print(f"Reading {file_path} (Session {session_type}, File {file_idx}/{len(files)})")
                
                try:
                    raw = self._read_raw(file_path)
                    raw = self._preprocess(raw)


                    if ch_names is None:
                        ch_names = raw.ch_names
                    
                    # Extract trials from this file
                    trials = self._extract_trials(raw)
                    data = raw.get_data()
                    
                    # Process each trial
                    for trial in trials:
                        onset_sample = int(trial['onset'] * self.target_sfreq)
                        end_sample = onset_sample + self.window_samples
                        
                        if end_sample <= data.shape[1]:
                            trial_data = data[:, onset_sample:end_sample]
                            
                            # Calculate absolute time within session
                            trial_start_time = session_time_offset + trial['onset']
                            
                            all_trials.append({
                                'data': trial_data,
                                'label': trial['label'],
                                'session_id': session_id,
                                'trial_id': trial_counter,
                                'onset_time': trial_start_time,
                            })
                            trial_counter += 1
                    
                    # Update session time offset (add file duration)
                    file_duration = data.shape[1] / self.target_sfreq
                    session_time_offset += file_duration
                    
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
            dataset_name="BCIC2A_4class",
            task_type="motor_imaginary",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
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

        print(f"Saved {output_path} ({len(all_trials)} trials)")
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
        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                import traceback
                traceback.print_exc()

        return output_paths


def build_bcic2a(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build BCIC-2A dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for BCICIV2ABuilder

    Returns:
        List of output file paths
    """
    builder = BCICIV2ABuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BCIC-2A HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (GDF or MAT)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="auto", choices=["auto", "gdf", "mat"], 
                       help="File format (auto-detect if not specified)")
    args = parser.parse_args()

    build_bcic2a(args.raw_data_dir, args.output_dir, args.subjects, file_format=args.format)