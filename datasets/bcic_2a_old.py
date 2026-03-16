"""
BCIC-2A Dataset Builder.

BCI Competition IV Dataset 2a: Motor Imagery.
- 9 subjects
- 2 sessions per subject (training + evaluation)
- 6 runs per session, 48 trials per run (288 trials per session)
- 4 motor imagery classes: left hand, right hand, feet, tongue
- https://physionet.org/content/sleep-edf/1.0.0/

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)

Data Validation:
- Automatically validate if each trial's amplitude is within reasonable range
- Trials exceeding the threshold will be skipped and recorded
- Validation statistics report will be displayed after processing completes

Data Range Checking:
Use the check_bcic2a_data_range.py script to check the data range of processed HDF5 files.
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
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


# BCIC-2A Dataset Configuration
BCIC2A_INFO = DatasetInfo(
    dataset_name="BCIC2A_4class",
    task_type=DatasetTaskType.MOTOR_IMAGINARY,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=4,
    category_list=["left", "right", "foot", "tongue"],
    sampling_rate=250.0,
    montage="10_20",
    channels=[
        'FZ',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'P1', 'PZ', 'P2',
        'POZ',
    ],
)

# Label mapping: annotation description -> class index
BCIC2A_LABEL_MAP = {
    '769': 0,  # left hand
    '770': 1,  # right hand
    '771': 2,  # feet
    '772': 3,  # tongue
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
    # This avoids misclassification due to occasional spikes or artifacts
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    
    # Fallback to max if percentile is too small (edge case)
    # But prefer percentile for robustness
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    # Also check median absolute deviation (MAD) as a sanity check
    # If MAD suggests different unit, use the more conservative (larger) estimate
    mad = np.median(abs_data)
    if mad > 0:
        # MAD typically ~0.67 * std for normal distribution
        # Use 3 * MAD as rough estimate of typical max amplitude
        mad_based_estimate = 3 * mad
        # Use the larger of the two estimates to be conservative
        max_amp = max(max_amp, mad_based_estimate)

    if max_amp > 1e-2:  # > 0.01, likely microvolts (µV)
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts (mV)
        return data / 1e3, "mV"
    else:  # likely already Volts (V)
        return data, "V"


class BCIC2ABuilder:
    """Builder for BCIC-2A dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        file_format: str = "mat",  # 'mat', 'set' for EEGLAB, 'gdf' for original
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,  # Amplitude threshold in µV
    ):
        """
        Initialize BCIC-2A builder.

        Args:
            raw_data_dir: Directory containing raw files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds
            stride_sec: Stride length in seconds
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency
            file_format: File format ('mat', 'set' for EEGLAB, or 'gdf' for original)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        # If output_dir already ends with "BCIC2A", don't append it again
        if output_path.name == "BCIC2A":
            self.output_dir = output_path
        else:
            self.output_dir = output_path / "BCIC2A"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 250.0  # Original sampling rate
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
        """Get list of subject IDs (1-9)."""
        return list(range(1, 10))

    def _find_files(self, subject_id: int) -> list[Path]:
        """
        Find all files for a subject.
        
        BCIC-2A file naming convention:
        - Training: A0{subject_id}T.{ext} (e.g., A01T.set, A01T.gdf)
        - Evaluation: A0{subject_id}E.{ext} (e.g., A01E.set, A01E.gdf)
        """
        files = []
        ext = f".{self.file_format}"

        # Search for files matching pattern A0{subject_id}*.{ext}
        # This matches both A01T.mat/A01T.set and A01E.mat/A01E.set (or .gdf)
        # For .mat files, also check raw_data subdirectory
        if self.file_format == "mat":
            # Check if raw_data subdirectory exists
            raw_data_subdir = self.raw_data_dir / "raw_data"
            if raw_data_subdir.exists():
                search_dir = raw_data_subdir
            else:
                search_dir = self.raw_data_dir
        else:
            search_dir = self.raw_data_dir

        # Check if directory exists
        if not search_dir.exists():
            print(f"⚠️  Warning: Data directory does not exist: {search_dir}")
            return files
        
        pattern = f"A0{subject_id}*{ext}"
        
        # Try both exact patterns and recursive search
        found_files = list(search_dir.rglob(pattern))
        
        # If no files found, try to list what's actually in the directory
        if not found_files:
            print(f"⚠️  No files found matching pattern: {pattern}")
            print(f"   Searching in: {search_dir}")
            
            # List all files with the extension to help debug
            all_ext_files = list(search_dir.rglob(f"*{ext}"))
            if all_ext_files:
                print(f"   Found {len(all_ext_files)} files with extension .{self.file_format}:")
                for f in sorted(all_ext_files)[:10]:  # Show first 10
                    print(f"     - {f.name}")
                if len(all_ext_files) > 10:
                    print(f"     ... and {len(all_ext_files) - 10} more files")
            else:
                print(f"   No files with extension .{self.file_format} found in directory")
                # List directory contents
                if search_dir.is_dir():
                    dir_contents = list(search_dir.iterdir())
                    if dir_contents:
                        print(f"   Directory contents:")
                        for item in sorted(dir_contents)[:10]:
                            print(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")
        else:
            files = sorted(found_files)
            print(f"   Found {len(files)} file(s) for subject {subject_id}")

        return files

    def _read_raw_mat(self, file_path: Path):
        """
        Read .mat file and convert to MNE Raw object.
        
        BCIC-2A .mat file structure:
        - data: structured array with runs
        - Each run has: X (EEG data), trial (trial indices), y (labels), fs (sampling rate)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")
        
        mat_data = scipy.io.loadmat(str(file_path), squeeze_me=False, struct_as_record=False)
        data_struct = mat_data['data']  # Shape: (1, n_runs)
        
        # Concatenate all runs
        all_runs_data = []
        all_trials = []
        all_labels = []
        sfreq = None
        
        cumulative_samples = 0  # Track cumulative samples across runs
        
        for run_idx in range(data_struct.shape[1]):
            # Access nested structure: data_struct[0, run_idx] is array containing mat_struct
            run_struct = data_struct[0, run_idx][0, 0]  # Get mat_struct object
            X = run_struct.X  # EEG data: (n_samples, n_channels)
            trial_indices = run_struct.trial  # Trial start indices
            y = run_struct.y  # Labels: (n_trials,)
            
            # Extract sampling rate (may be nested array like [[250]])
            fs_array = run_struct.fs
            if isinstance(fs_array, np.ndarray):
                fs = float(fs_array.item() if fs_array.size == 1 else fs_array[0, 0])
            else:
                fs = float(fs_array)
            
            if sfreq is None:
                sfreq = fs
            elif sfreq != fs:
                print(f"  Warning: Sampling rate mismatch in run {run_idx}: {fs} vs {sfreq}")
            
            # Transpose to (n_channels, n_samples) for MNE
            X = X.T
            
            # Store run data
            all_runs_data.append(X)
            
            # Process trials if they exist
            if trial_indices.size > 0 and y.size > 0:
                # Convert trial indices to onset times (relative to start of concatenated data)
                trial_indices_flat = trial_indices.flatten()
                y_flat = y.flatten()
                
                # Trial times are relative to the start of this run, need to add cumulative offset
                trial_times = (cumulative_samples + trial_indices_flat) / sfreq
                
                for trial_time, label in zip(trial_times, y_flat):
                    all_trials.append({
                        'onset': float(trial_time),
                        'label': int(label) - 1,  # Convert 1-4 to 0-3
                        'run': run_idx,
                    })
            
            # Update cumulative samples for next run
            cumulative_samples += X.shape[1]
        
        # Concatenate all runs
        if all_runs_data:
            concatenated_data = np.concatenate(all_runs_data, axis=1)
        else:
            raise ValueError("No data found in .mat file")
        
        # If original file有25个通道（22 EEG + 3 EOG），这里显式丢弃眼动，只保留22个EEG
        n_channels = concatenated_data.shape[0]
        if n_channels >= 25 and n_channels != len(BCIC2A_INFO.channels):
            print(
                f"  Detected {n_channels} channels, assuming first "
                f"{len(BCIC2A_INFO.channels)} are EEG and dropping the rest (EOG)."
            )
            concatenated_data = concatenated_data[: len(BCIC2A_INFO.channels), :]
            n_channels = concatenated_data.shape[0]

        # Create MNE Info object
        if n_channels == len(BCIC2A_INFO.channels):
            # 标准情况：22个EEG通道，使用规范通道名
            ch_names = BCIC2A_INFO.channels
        else:
            # 非标准情况：退回占位名并给出明显警告
            print(
                f"  Warning: expected {len(BCIC2A_INFO.channels)} EEG channels for BCIC2A, "
                f"but got {n_channels}. Using generic EEG01.. names."
            )
            ch_names = [f'EEG{i+1:02d}' for i in range(n_channels)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels
        )
        
        # Convert data to Volts (assuming data is in µV)
        # Auto-detect unit and convert
        data_volts, detected_unit = detect_unit_and_convert_to_volts(concatenated_data)
        if detected_unit != "V":
            print(f"  Detected unit: {detected_unit}, converted to V")
        
        # Create MNE Raw object
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        
        # Add annotations for trials
        # Convert labels to annotation descriptions (769-772 for left, right, feet, tongue)
        if all_trials:
            onsets = [t['onset'] for t in all_trials]
            descriptions = [str(769 + t['label']) for t in all_trials]  # 769-772
            raw.set_annotations(mne.Annotations(onsets, [4.0] * len(onsets), descriptions))
        
        return raw

    def _read_raw(self, file_path: Path):
        """
        Read raw EEG file and ensure data is in Volts.

        MNE may preserve the original unit from the file, so we detect and convert
        to Volts to ensure consistent processing.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if self.file_format == "mat":
                raw = self._read_raw_mat(file_path)
            elif self.file_format == "set":
                raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
            elif self.file_format == "gdf":
                raw = mne.io.read_raw_gdf(str(file_path), preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

        # Auto-detect unit and convert to Volts for MNE processing (for set/gdf files)
        # MNE uses Volts internally, so we need to ensure data is in V
        if self.file_format in ["set", "gdf"]:
            if hasattr(raw, '_data') and raw._data is not None:
                max_amp = np.abs(raw._data).max()
                data_volts, detected_unit = detect_unit_and_convert_to_volts(raw._data)
                
                if detected_unit != "V":
                    raw._data = data_volts
                    print(f"  Detected unit: {detected_unit}, converted to V (max amplitude: {max_amp:.2e} {detected_unit})")

        # 无论是 mat 还是 set/gdf，这里统一只保留 EEG 通道，丢弃眼动/多余通道
        if len(raw.ch_names) > len(BCIC2A_INFO.channels):
            # 优先按通道名挑选标准EEG通道，若失败则按索引取前22个
            try:
                raw.pick(BCIC2A_INFO.channels)
                print(f"  Picked EEG channels by name: {BCIC2A_INFO.channels}")
            except Exception:
                raw.pick(raw.ch_names[: len(BCIC2A_INFO.channels)])
                print(
                    f"  Warning: could not pick by canonical names, "
                    f"kept first {len(BCIC2A_INFO.channels)} channels and dropped the rest as EOG."
                )

        return raw

    def _preprocess(self, raw) -> np.ndarray:
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
            subject_id: Subject identifier (1-9)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building BCIC-2A dataset")

        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        all_trials = []
        ch_names = None

        # Process each file (session)
        for session_id, file_path in enumerate(files, 1):
            print(f"Reading {file_path}")

            raw = self._read_raw(file_path)
            raw = self._preprocess(raw)

            if ch_names is None:
                ch_names = raw.ch_names

            # Extract trials
            trials = self._extract_trials(raw)
            data = raw.get_data()

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
                        print(f"  Skipping trial {len(all_trials)}: amplitude {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                        self.rejected_trials += 1
                        continue

                    self.valid_trials += 1
                    all_trials.append({
                        'data': trial_data_uv,  # Store in µV
                        'label': trial['label'],
                        'session_id': session_id,
                        'trial_id': len(all_trials),
                        'onset_time': trial['onset'],  # Store onset time in session
                    })

        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="BCIC2A_4class",
            task_type="motor_imaginary",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=BCIC2A_INFO.num_labels,
            category_list=BCIC2A_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
        )

        # Create output file
        # Check if output_dir exists and is a file (not a directory)
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
                # Time is relative to session start
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
                "name": BCIC2A_INFO.dataset_name,
                "description": "BCI Competition IV Dataset 2a: Motor Imagery",
                "task_type": str(BCIC2A_INFO.task_type.value),
                "downstream_task": str(BCIC2A_INFO.downstream_task_type.value),
                "num_labels": BCIC2A_INFO.num_labels,
                "category_list": BCIC2A_INFO.category_list,
                "original_sampling_rate": BCIC2A_INFO.sampling_rate,
                "channels": BCIC2A_INFO.channels,
                "montage": BCIC2A_INFO.montage,
                "source_url": "https://www.bbci.de/competition/iv/",
                "num_subjects": 9,
                "num_sessions_per_subject": 2,
                "num_runs_per_session": 6,
                "num_trials_per_run": 48,
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

        # Ensure output directory exists before saving JSON
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


def build_bcic2a(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build BCIC2A dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for BCIC2ABuilder

    Returns:
        List of output file paths
    """
    builder = BCIC2ABuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BCIC2A HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to process")
    parser.add_argument("--format", default="mat", choices=["mat", "set", "gdf"], help="File format")
    args = parser.parse_args()

    build_bcic2a(args.raw_data_dir, args.output_dir, args.subjects, file_format=args.format)
