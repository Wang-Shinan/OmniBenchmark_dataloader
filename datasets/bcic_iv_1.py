"""
BCI Competition IV Dataset 1 (BCICIV-1) Dataset Builder.

BCICIV-1: Motor Imagery Dataset
- 7 subjects (a-g)
- 2 sessions per subject (calibration + evaluation)
- 2 motor imagery classes: left hand, foot
- 100 Hz sampling rate (or 1000 Hz in 1000Hz version)
- 59 channels (10-10 montage)
- https://www.bbci.de/competition/iv/

Data Structure:
- .mat files contain:
  - cnt: continuous EEG data (n_samples x n_channels), int16
  - nfo: info structure with fs (sampling rate), classes, clab (channel names), xpos, ypos
  - mrk: markers structure with pos (event positions) and y (labels) - only in calib files

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Data is in int16 format, needs conversion to voltage units
- Automatically detect unit and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5
- Default amplitude validation threshold: 600 µV
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


# BCICIV-1 Dataset Configuration
BCICIV1_INFO = DatasetInfo(
    dataset_name="BCICIV1_2class",
    task_type=DatasetTaskType.MOTOR_IMAGINARY,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["left", "right"],
    sampling_rate=1000.0,  # Original sampling rate (1000Hz version also available)
    montage="10_10",
    channels=[
        'AF3', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        'CFC7', 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
        'PO1', 'PO2', 'O1', 'O2',
    ],
)

# Label mapping: marker value -> class index
# According to BCICIV1 documentation:
# y = -1 for class one (first class in nfo.classes)
# y = 1 for class two (second class in nfo.classes)
# Each subject selects 2 classes from: left hand, right hand, foot
# So the mapping is dynamic based on nfo.classes for each subject
# We map: y=-1 -> label 0 (first class), y=1 -> label 1 (second class)
BCICIV1_LABEL_MAP = {
    -1: 0,  # class one -> label 0
    1: 1,   # class two -> label 1
}

# Channels to remove (if any reference channels)
REMOVE_CHANNELS = []

# Default amplitude threshold (µV) for validation
# Based on data analysis:
# - 95th percentile: ~3772 µV
# - 99th percentile: ~4730 µV
# - Max (excluding outlier): ~6000 µV
# - Subject f has one outlier at 22025 µV (likely artifact)
# Recommended: 5000-6000 µV to reject only clear artifacts while keeping valid data
DEFAULT_MAX_AMPLITUDE_UV = 6000.0


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


class BCICIV1Builder:
    """Builder for BCICIV-1 dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 50Hz for Europe/Asia
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        trial_duration_sec: float = 4.0,  # Duration of each trial in seconds
        clean_mode: bool = False,  # If True, filter out foot trials from subjects a and f to create pure left-right classification
    ):
        """
        Initialize BCICIV-1 builder.

        Args:
            raw_data_dir: Directory containing raw files
            output_dir: Output directory for HDF5 files
            target_sfreq: Target sampling frequency
            window_sec: Window length in seconds for segmentation
            stride_sec: Stride length in seconds for segmentation
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            filter_notch: Notch filter frequency (50Hz for Europe/Asia, 60Hz for Americas)
            max_amplitude_uv: Maximum amplitude threshold in µV for validation
            trial_duration_sec: Duration of each trial in seconds (default 4.0)
            clean_mode: If True, filter out foot (label=1) trials from subjects a and f
                        to create a pure left-right binary classification task
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.clean_mode = clean_mode
        output_path = Path(output_dir)
        if clean_mode:
            if output_path.name == "BCICIV1_clean":
                self.output_dir = output_path
            else:
                self.output_dir = output_path / "BCICIV1_clean"
        else:
            if output_path.name == "BCICIV1":
                self.output_dir = output_path
            else:
                self.output_dir = output_path / "BCICIV1"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 1000.0  # Will be read from file
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.trial_duration_sec = trial_duration_sec

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.trial_samples = int(trial_duration_sec * target_sfreq)

        # Track validation statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs (a-g)."""
        return ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def _find_files(self, subject_id: str) -> dict[str, Path]:
        """
        Find calibration and evaluation files for a subject.
        
        File naming: BCICIV_calib_ds1{subject_id}.mat, BCICIV_eval_ds1{subject_id}.mat
        """
        files = {}
        
        calib_file = self.raw_data_dir / f"BCICIV_calib_ds1{subject_id}_1000Hz.mat"
        eval_file = self.raw_data_dir / f"BCICIV_eval_ds1{subject_id}_1000Hz.mat"
        
        if calib_file.exists():
            files['calibration'] = calib_file
        else:
            print(f"  Warning: Calibration file not found: {calib_file}")
            
        if eval_file.exists():
            files['evaluation'] = eval_file
        else:
            print(f"  Warning: Evaluation file not found: {eval_file}")
            
        return files

    def _read_raw_mat(self, file_path: Path, has_labels: bool = True):
        """
        Read .mat file and convert to MNE Raw object.
        
        BCICIV-1 .mat file structure:
        - cnt: continuous EEG data (n_samples x n_channels), int16
        - nfo: info structure with fs, classes, clab (channel names), xpos, ypos
        - mrk: markers structure with pos (event positions) and y (labels) - only in calib files
        
        Args:
            file_path: Path to .mat file
            has_labels: Whether the file contains labels (True for calib, False for eval)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required for reading .mat files")
        
        # Load .mat file with error handling for corrupt files
        try:
            mat_data = scipy.io.loadmat(str(file_path), struct_as_record=False, squeeze_me=False)
        except Exception as e:
            if "corrupt" in str(e).lower() or "first" in str(e).lower():
                raise FileNotFoundError(f"Mat file appears to be corrupt or empty: {file_path}") from e
            else:
                raise
        
        # Extract data
        cnt = mat_data['cnt']  # Shape: (n_samples, n_channels)
        nfo = mat_data['nfo'][0, 0]  # Info structure
        
        # Get sampling rate
        fs = float(nfo.fs[0, 0])
        self.orig_sfreq = fs
        
        # Get channel names
        clab = nfo.clab[0]  # Array of channel name arrays
        ch_names = [str(ch[0]) for ch in clab]
        
        # Transpose to (n_channels, n_samples) for MNE
        data = cnt.T.astype(np.float64)
        
        # Convert from int16 to voltage units
        # Typical EEG data in int16 format: need to scale appropriately
        # Assuming data is in microvolts, but we'll auto-detect
        # Common scaling: if max is around 10000-50000, likely µV
        max_val = np.abs(data).max()
        if max_val > 1000:  # Likely needs scaling
            # Try to detect if it's already in reasonable µV range
            # If max is > 10000, might need division
            if max_val > 10000:
                # Could be raw ADC values, need to scale
                # For now, assume it's in µV if reasonable, otherwise scale
                data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
            else:
                # Assume already in µV
                data_volts = data / 1e6
                detected_unit = "µV"
        else:
            data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        
        print(f"  Detected unit: {detected_unit}, max amplitude: {max_val:.2e}")
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=['eeg'] * len(ch_names)
        )
        
        # Create MNE Raw object
        raw = mne.io.RawArray(data_volts, info, verbose=False)
        
        # Add annotations for trials (only if labels available)
        if has_labels and 'mrk' in mat_data:
            mrk = mat_data['mrk'][0, 0]
            pos = mrk.pos[0]  # Event positions (sample indices)
            # According to BCICIV1 documentation:
            # y = -1 for class one (left), y = 1 for class two (foot)
            y = mrk.y[0]  # Labels (-1=class one/left, 1=class two/foot)
            
            # Get class names from nfo
            classes = [str(c[0]) for c in nfo.classes[0]]
            
            # Filter out invalid trials (only keep y=-1 or y=1)
            # Note: According to documentation, y=-1 and y=1 are both valid class labels
            valid_mask = np.isin(y, [-1, 1])
            pos_valid = pos[valid_mask]
            y_valid = y[valid_mask]
            
            if len(pos_valid) > 0:
                # Convert sample positions to time (seconds)
                onsets = []
                descriptions = []
                
                # Convert labels: -1->0 (class one/left), 1->1 (class two/foot)
                for i, label_val in enumerate(y_valid):
                    if label_val in BCICIV1_LABEL_MAP:
                        mapped_label = BCICIV1_LABEL_MAP[label_val]
                        onsets.append(pos_valid[i] / fs)
                        descriptions.append(str(mapped_label))
                    else:
                        # Skip invalid labels (should not happen if valid_mask is correct)
                        print(f"  Warning: Unexpected label value {label_val}, skipping")
                        continue
                
                if len(onsets) > 0:
                    # Add annotations with trial duration
                    durations = [self.trial_duration_sec] * len(onsets)
                    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
            else:
                print(f"  Warning: No valid trials found (all trials have label -1)")
        
        return raw

    def _read_raw(self, file_path: Path, has_labels: bool = True):
        """Read raw EEG file and convert to MNE Raw object."""
        if not HAS_MNE:
            raise ImportError("MNE is required")
        
        if file_path.suffix == '.mat':
            return self._read_raw_mat(file_path, has_labels)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """
        Validate trial amplitude.

        Args:
            trial_data: Trial data in µV (shape: n_channels x n_samples)

        Returns:
            True if amplitude is within threshold, False otherwise
        """
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report trial validation statistics."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Valid trials: {self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  Rejected trials: {self.rejected_trials} ({100-valid_pct:.1f}%)")

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        # Drop reference channels if needed
        if REMOVE_CHANNELS:
            channels_to_drop = [ch for ch in REMOVE_CHANNELS if ch in raw.ch_names]
            if channels_to_drop:
                raw.drop_channels(channels_to_drop)

        # Resample first if needed (before filtering to avoid Nyquist issues)
        # For low sampling rates (e.g., 100Hz), resample before notch filter
        if raw.info['sfreq'] != self.target_sfreq and raw.info['sfreq'] < 150:
            raw.resample(self.target_sfreq, verbose=False)

        # Notch filter (only if notch frequency is less than Nyquist)
        if self.filter_notch > 0:
            nyquist = raw.info['sfreq'] / 2.0
            if self.filter_notch < nyquist:
                raw.notch_filter(freqs=self.filter_notch, verbose=False)
            else:
                print(f"  Warning: Skipping notch filter ({self.filter_notch}Hz) - exceeds Nyquist ({nyquist:.1f}Hz)")

        # Bandpass filter
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)

        # Resample if still needed (for high sampling rates)
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        return raw

    def _extract_trials(self, raw, session_name: str) -> list[dict]:
        """
        Extract trials from raw data.
        
        Args:
            raw: MNE Raw object
            session_name: Name of the session ('calibration' or 'evaluation')
        """
        trials = []
        
        # Extract from annotations (only calib files have annotations)
        if raw.annotations and len(raw.annotations) > 0:
            for onset, duration, desc in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description
            ):
                # Map description to label
                try:
                    label = int(desc)
                except ValueError:
                    label = 0  # Default to class 0
                
                trials.append({
                    'onset': onset,
                    'duration': duration,
                    'label': label,
                })
        else:
            # Evaluation files don't have labels
            # Could extract segments without labels, but for now we skip eval files
            print(f"  No annotations found in {session_name} file, skipping trial extraction")
        
        return trials

    def build_subject(self, subject_id: str) -> str:
        """
        Build HDF5 file for a single subject.

        Args:
            subject_id: Subject identifier (a-g)

        Returns:
            Path to output HDF5 file
        """
        if not HAS_MNE:
            raise ImportError("MNE is required for building BCICIV-1 dataset")

        # Reset validation counters
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # Find files
        files = self._find_files(subject_id)
        if not files:
            raise FileNotFoundError(f"No files found for subject {subject_id}")
        
        # Check if we have at least one calib file (required for labels)
        calib_files = [f for name, f in files.items() if name == 'calibration']
        if not calib_files:
            raise FileNotFoundError(f"No calibration file found for subject {subject_id} (calibration file required for labels)")

        all_trials = []
        ch_names = None
        category_list = None  # Will be read from calibration file

        # First, read calibration file to get category_list
        calib_file_path = calib_files[0]
        if HAS_SCIPY:
            try:
                mat_data = scipy.io.loadmat(str(calib_file_path), struct_as_record=False, squeeze_me=False)
                nfo = mat_data['nfo'][0, 0]
                category_list = [str(c[0]) for c in nfo.classes[0]]
                print(f"  Detected classes for subject {subject_id}: {category_list}")
            except Exception as e:
                print(f"  Warning: Could not read classes from calibration file: {e}")

        # Process each session (calibration has labels, evaluation doesn't)
        for session_name, file_path in files.items():
            print(f"Reading {file_path} (session: {session_name})")
            
            # Only calib files have labels
            has_labels = (session_name == 'calibration')
            
            try:
                raw = self._read_raw(file_path, has_labels=has_labels)
                raw = self._preprocess(raw)

                if ch_names is None:
                    ch_names = raw.ch_names

                # Extract trials (only from calib files)
                if has_labels:
                    trials = self._extract_trials(raw, session_name)
                    data = raw.get_data()  # In Volts
                    current_sfreq = raw.info['sfreq']  # Use actual sampling rate after preprocessing

                    if not trials:
                        print(f"  Warning: No trials extracted from {session_name} file")
                        continue

                    # Process each trial
                    for trial_idx, trial in enumerate(trials):
                        # Use current sampling rate (after resampling) to calculate sample indices
                        onset_sample = int(trial['onset'] * current_sfreq)
                        trial_samples_current = int(self.trial_duration_sec * current_sfreq)
                        end_sample = onset_sample + trial_samples_current

                        if end_sample <= data.shape[1] and onset_sample >= 0:
                            # Convert from V to µV
                            trial_data_v = data[:, onset_sample:end_sample]
                            trial_data_uv = trial_data_v * 1e6

                            # Validate trial amplitude
                            self.total_trials += 1
                            if not self._validate_trial(trial_data_uv):
                                max_amp = np.abs(trial_data_uv).max()
                                print(f"  Skipping trial {trial_idx}: amplitude {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                                self.rejected_trials += 1
                                continue

                            # In clean_mode, filter out foot (label=1) trials from subjects a and f
                            if self.clean_mode and subject_id in ['a', 'f'] and trial['label'] == 1:
                                print(f"  Skipping trial {trial_idx}: foot trial (label=1) filtered in clean_mode")
                                self.rejected_trials += 1
                                continue
                            
                            self.valid_trials += 1
                            all_trials.append({
                                'data': trial_data_uv,
                                'label': trial['label'],
                                'trial_id': len(all_trials),
                                'session_id': 1 if session_name == 'calibration' else 2,
                                'onset_time': trial['onset'],
                            })
                        else:
                            print(f"  Warning: Trial {trial_idx} out of bounds (onset={onset_sample}, end={end_sample}, data_len={data.shape[1]})")
                else:
                    print(f"  Skipping {session_name} file (no labels available)")
                    
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                # Continue processing other files even if one fails
                continue

        if not all_trials:
            error_msg = f"No valid trials extracted for subject {subject_id}"
            if self.total_trials > 0:
                error_msg += f" ({self.total_trials} trials processed, {self.rejected_trials} rejected)"
            raise ValueError(error_msg)

        # In clean_mode, use ['left', 'right'] for all subjects
        # Note: foot trials (label=1) from subjects a and f are already filtered above
        if self.clean_mode:
            category_list = ['left', 'right']
            print(f"  Clean mode: Using unified category_list: {category_list}")
        else:
            # Use detected category_list if available, otherwise fall back to default
            if category_list is None:
                category_list = BCICIV1_INFO.category_list
                print(f"  Warning: Using default category_list: {category_list}")
        
        # Ensure num_labels matches category_list length
        num_labels = len(category_list) if category_list else BCICIV1_INFO.num_labels
        
        # Determine dataset name based on clean_mode
        dataset_name = "BCICIV1_clean_2class" if self.clean_mode else BCICIV1_INFO.dataset_name
        
        # Create subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=dataset_name,
            task_type=BCICIV1_INFO.task_type.value,
            downstream_task_type=BCICIV1_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=num_labels,
            category_list=category_list,  # Use detected classes (or ['left', 'right'] in clean_mode)
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=BCICIV1_INFO.montage,
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

                # Segment trial into windows
                trial_data = trial['data']  # Shape: (n_channels, n_samples)
                n_samples = trial_data.shape[1]

                for i_slice, start in enumerate(range(0, n_samples - self.window_samples + 1, self.stride_samples)):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=trial['onset_time'] + start / self.target_sfreq,
                        end_time=trial['onset_time'] + end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data)

        # Report validation statistics
        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        dataset_name = "BCICIV1_clean_2class" if self.clean_mode else BCICIV1_INFO.dataset_name
        description = "BCI Competition IV Dataset 1: Motor Imagery (Left Hand vs Right Hand, Clean Version)" if self.clean_mode else "BCI Competition IV Dataset 1: Motor Imagery (Left Hand vs Foot)"
        category_list = ['left', 'right'] if self.clean_mode else BCICIV1_INFO.category_list
        
        info = {
            "dataset": {
                "name": dataset_name,
                "description": description,
                "task_type": str(BCICIV1_INFO.task_type.value),
                "downstream_task": str(BCICIV1_INFO.downstream_task_type.value),
                "num_labels": 2,
                "category_list": category_list,
                "original_sampling_rate": BCICIV1_INFO.sampling_rate,
                "channels": BCICIV1_INFO.channels,
                "montage": BCICIV1_INFO.montage,
                "source_url": "https://www.bbci.de/competition/iv/",
                "num_subjects": 7,
                "subjects": ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                "clean_mode": self.clean_mode,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "trial_duration_sec": self.trial_duration_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "clean_mode": self.clean_mode,
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
        with open(json_path, "w", encoding="utf-8") as f:
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


def build_bciciv1(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    **kwargs,
) -> list[str]:
    """
    Convenience function to build BCICIV-1 dataset.

    Args:
        raw_data_dir: Directory containing raw files
        output_dir: Output directory for HDF5 files
        subject_ids: List of subject IDs to process (None = all)
        **kwargs: Additional arguments for BCICIV1Builder

    Returns:
        List of output file paths
    """
    builder = BCICIV1Builder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BCICIV-1 HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", type=str, help="Subject IDs to process (a-g)")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling frequency")
    parser.add_argument("--window_sec", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Stride length in seconds")
    parser.add_argument("--trial_duration_sec", type=float, default=4.0, help="Trial duration in seconds")
    parser.add_argument("--clean_mode", action="store_true", help="Filter out foot trials from subjects a and f to create pure left-right classification")
    args = parser.parse_args()

    build_bciciv1(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        trial_duration_sec=args.trial_duration_sec,
        clean_mode=args.clean_mode,
    )
