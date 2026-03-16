"""
PerceiveImagine EEG Dataset Builder.

PerceiveImagine EEG Dataset
- Multiple subjects
- Task: Perceive vs Imagine (binary classification)
- EEGLAB format (.set files)
- BIDS-compatible structure

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)

Label Mapping:
- Original label value 1 -> class index 0 (perceive)
- Original label value 2 -> class index 1 (imagine)
- Original label value 255 -> invalid label, will be filtered
"""

from pathlib import Path
import argparse
import json
import numpy as np
import re
from datetime import datetime
import warnings

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import h5py
except ImportError:
    raise SystemExit('h5py is required')

import pandas as pd

# ===================== Dataset Configuration =====================
class DatasetTaskType:
    """Task type enumeration"""
    RESTING_STATE = "resting_state"
    OTHER = "other"

class DatasetInfo:
    """Dataset core information configuration"""
    def __init__(self):
        self.dataset_name = "PerceiveImagine_EEG_Dataset"
        self.task_type = DatasetTaskType.OTHER  # Task state (Perceive/Imagine)
        self.downstream_task_type = "classification"  # Binary classification task
        self.montage = "10_20"  # Consistent with EEGPlacementScheme in JSON file
        self.num_labels = 2  # Binary classification: perceive vs imagine
        self.category_list = ["perceive", "imagine"]  # Category list

# Global dataset info instance
PERCEIVE_IMAGINE_INFO = DatasetInfo()

# ===================== Constants Configuration =====================
# Channels to remove (non-EEG channels)
REMOVE_CHANNELS = ['TRIGGER']

# Default preprocessing parameters
DEFAULT_MAX_AMPLITUDE_UV = 600.0  # Default amplitude validation threshold (µV)
DEFAULT_TARGET_SFREQ = 200.0  # Target sampling rate (Hz)
DEFAULT_FILTER_LOW = 0.5  # Low cutoff frequency for bandpass filter (Hz)
DEFAULT_FILTER_HIGH = 70.0  # High cutoff frequency for bandpass filter (Hz)
DEFAULT_FILTER_NOTCH = 50.0  # Notch filter frequency (Hz, China uses 50 Hz)
DEFAULT_WINDOW_SEC = 1.0  # Window length (seconds)
DEFAULT_STRIDE_SEC = 1.0  # Window stride (seconds)

# Trial-related parameters
TRIAL_DURATION_SEC = 6.0  # Trial duration (seconds)
TRIAL_RESERVE_SEC = 7.0  # Trial reserve time (seconds, includes buffer)

# Reference channels (empty list means using average reference)
REFERENCE_CHANNELS = []

# Label mapping: original event value -> (class_index, class_name)
# Class index is used for HDF5 storage, class name is for readability
LABEL_MAPPING = {
    1: (0, "perceive"),  # Perceive task -> class 0
    2: (1, "imagine"),   # Imagine task -> class 1
    255: (None, "n/a")   # Invalid label, will be filtered
}

# Valid label values (excluding 255)
VALID_LABEL_VALUES = [k for k, v in LABEL_MAPPING.items() if v[0] is not None]

# Category list (consistent with category_list in DatasetInfo)
CATEGORY_LIST = ["perceive", "imagine"]

# ===================== Data Classes =====================
class SubjectAttrs:
    """Subject-level metadata"""
    def __init__(
        self,
        subject_id: int,
        dataset_name: str,
        task_type: str,
        downstream_task_type: str,
        rsFreq: float,
        chn_name: list,
        num_labels: int,
        category_list: list,
        chn_pos: list | None = None,  # Channel positions (default: None)
        chn_ori: None = None,
        chn_type: str = "EEG",
        montage: str = "10_20"
    ):
        self.subject_id = subject_id
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.downstream_task_type = downstream_task_type
        self.rsFreq = rsFreq
        self.chn_name = chn_name
        self.num_labels = num_labels
        self.category_list = category_list
        
        # Handle chn_pos: set to None if empty list or None
        self.chn_pos = chn_pos if (chn_pos is not None and len(chn_pos) > 0) else None
        self.chn_ori = chn_ori
        
        self.chn_type = chn_type
        self.montage = montage

    def to_dict(self) -> dict:
        """Convert to dictionary for HDF5 storage"""
        # Handle chn_pos conversion: convert list to string array or "None"
        chn_pos_val = self.chn_pos
        if isinstance(chn_pos_val, list):
            # List type: convert to HDF5-compatible string array
            dt = h5py.string_dtype(encoding='utf-8')
            chn_pos_pure_str = [str(item).strip() if item is not None else " " for item in chn_pos_val]
            chn_pos_val = np.array(chn_pos_pure_str, dtype=dt)
        elif chn_pos_val is None:
            # None value: convert to "None" string to avoid HDF5 type errors
            chn_pos_val = "None"

        # Handle chn_ori: convert None to "None" string
        chn_ori_val = "None" if self.chn_ori is None else self.chn_ori

        return {
            'subject_id': self.subject_id,
            'dataset_name': self.dataset_name,
            'task_type': self.task_type,
            'downstream_task_type': self.downstream_task_type,
            'rsFreq': self.rsFreq,
            'chn_name': self.chn_name,
            'num_labels': self.num_labels,
            'category_list': self.category_list,
            'chn_pos': chn_pos_val,
            'chn_ori': chn_ori_val,
            'chn_type': self.chn_type,
            'montage': self.montage
        }

class TrialAttrs:
    """Trial-level metadata"""
    def __init__(self, trial_id: int, session_id: int):
        self.trial_id = trial_id
        self.session_id = session_id

    def to_dict(self) -> dict:
        return {
            'trial_id': self.trial_id,
            'session_id': self.session_id
        }

class SegmentAttrs:
    """Segment-level metadata
    
    Note: The label field should use a numeric array (numpy.ndarray), not a string.
    Format: np.array([label_index]), where label_index is 0 or 1.
    """
    def __init__(
        self,
        segment_id: int,
        start_time: float,
        end_time: float,
        time_length: float,
        label: np.ndarray  # 数字标签数组，如 np.array([0]) 或 np.array([1])
    ):
        self.segment_id = segment_id
        self.start_time = start_time
        self.end_time = end_time
        self.time_length = time_length
        self.label = label

    def to_dict(self) -> dict:
        return {
            'segment_id': self.segment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'time_length': self.time_length,
            'label': self.label
        }

# ===================== Utility Functions =====================
def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect data unit and convert to Volts for MNE.
    
    MNE internally uses Volts (V) as the unit, so we need to ensure data is in V before processing.
    This function detects the unit based on amplitude range and converts accordingly.
    
    Uses robust statistics (percentile) instead of max to avoid noise/artifact interference.
    
    Args:
        data: Input data array (shape: n_channels x n_samples)
    
    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    
    # Use median absolute deviation (MAD) as additional check
    mad = np.median(abs_data)
    if mad > 0:
        mad_based_estimate = 3 * mad
        max_amp = max(max_amp, mad_based_estimate)
    
    if max_amp > 1e-2:  # > 0.01, likely microvolts (µV)
        return data / 1e6, 'µV'
    elif max_amp > 1e-5:  # > 0.00001, likely millivolts (mV)
        return data / 1e3, 'mV'
    else:  # likely already Volts (V)
        return data, 'V'

def is_non_eeg_channel(name: str) -> bool:
    """
    Check if channel name is a non-EEG channel.
    
    Args:
        name: Channel name
    
    Returns:
        bool: True if non-EEG channel, False otherwise
    """
    un = name.upper()
    for s in REMOVE_CHANNELS:
        if s in un:
            return True
    return False

def clean_channel_name(name: str) -> str:
    """
    Clean channel name: strip whitespace and convert to uppercase.
    
    Args:
        name: Original channel name
    
    Returns:
        str: Cleaned channel name
    """
    s = name.strip()
    s = s.upper()
    return s

def parse_set_filename(filename: str) -> dict:
    """
    Parse .set filename to extract subject ID and task information.
    
    Args:
        filename: Filename
    
    Returns:
        dict: Dictionary containing subject, task, session, trial, valid fields
    """
    stem = Path(filename).stem
    pattern = r"sub-(?P<subject>\d+)_task-(?P<task>\w+)_eeg"
    m = re.search(pattern, stem, re.IGNORECASE)
    if m:
        return {
            'subject': m.group('subject').strip(),
            'task': m.group('task').strip(),
            'session': 1,
            'trial': 0,
            'valid': True
        }
    else:
        return {
            'subject': '01',
            'task': 'PerceiveImagine',
            'session': 1,
            'trial': 0,
            'valid': False
        }

def map_label_value_to_index(value: int) -> tuple[int | None, str]:
    """
    Map original label value to class index and class name.
    
    Args:
        value: Original event label value (1, 2, or 255)
    
    Returns:
        tuple: (label_index, label_name)
            - label_index: Class index (0 or 1), None if invalid
            - label_name: Class name ("perceive" or "imagine"), "n/a" if invalid
    """
    return LABEL_MAPPING.get(value, (None, "n/a"))

def load_sub01_aux_files(set_file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, list, list]:
    """
    Load auxiliary files (channels.tsv, events.tsv, JSON metadata).
    
    This function will:
    1. Find and load channels.tsv (channel information)
    2. Find and load events.tsv (event/label information)
    3. Find and load JSON metadata file
    4. Process channel names, remove non-EEG channels
    5. Process event labels, map original values to class indices and names, filter invalid labels
    
    Args:
        set_file_path: Path to .set file
    
    Returns:
        tuple: (channels_df, events_df, eeg_metadata, channel_names, channel_positions)
            - channels_df: Channel information DataFrame
            - events_df: Event information DataFrame (contains label_index and label_name columns)
            - eeg_metadata: JSON metadata dictionary
            - channel_names: Cleaned channel names list
            - channel_positions: Channel positions list (placeholder, will be populated from montage)
    
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If file format is incorrect
    """
    base_dir = set_file_path.parent
    
    # Search for channels.tsv (support multiple case variants)
    channels_files = list(base_dir.glob("*channels*.tsv"))
    if not channels_files:
        channels_files = list(base_dir.glob("*Channels*.tsv"))
    if not channels_files:
        channels_files = list(base_dir.glob("*CHANNELS*.tsv"))
    if not channels_files:
        raise FileNotFoundError(f"Channels .tsv file not found in {base_dir}")
    channels_file = channels_files[0]
    
    # Search for events.tsv (support multiple case variants)
    events_files = list(base_dir.glob("*events*.tsv"))
    if not events_files:
        events_files = list(base_dir.glob("*Events*.tsv"))
    if not events_files:
        events_files = list(base_dir.glob("*EVENTS*.tsv"))
    if not events_files:
        raise FileNotFoundError(f"Events .tsv file not found in {base_dir}")
    events_file = events_files[0]
    
    # Search for JSON metadata
    json_files = list(base_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON metadata file not found in {base_dir}")
    json_file = json_files[0]
    
    # Load and process files
    channels_df = pd.read_csv(channels_file, sep='\t')
    events_df = pd.read_csv(events_file, sep='\t')
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            eeg_metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file format: {str(e)}")
    
    # Extract channel names
    if "name" not in channels_df.columns:
        raise ValueError("Channel file missing 'name' column")
    channel_names = [clean_channel_name(name) for name in channels_df["name"].tolist()]
    channel_names = [name for name in channel_names if not is_non_eeg_channel(name)]
    
    # Label mapping and filtering
    if "value" not in events_df.columns:
        raise ValueError("Event file missing 'value' column")
    
    # Map original label values to class indices and names
    label_mappings = events_df['value'].apply(map_label_value_to_index)
    events_df['label_index'] = label_mappings.apply(lambda x: x[0])
    events_df['label_name'] = label_mappings.apply(lambda x: x[1])
    
    # Filter invalid labels (rows where label_index is None)
    events_df = events_df[events_df['label_index'].notna()].reset_index(drop=True)
    
    # Ensure label_index is integer type
    events_df['label_index'] = events_df['label_index'].astype(int)
    
    # Remove unnecessary columns (if present)
    if "sample" in events_df.columns:
        events_df = events_df.drop(columns=["sample"])
    
    # Channel positions initialization (placeholder, will be populated from montage)
    channel_positions = [" " for _ in channel_names]
    
    return channels_df, events_df, eeg_metadata, channel_names, channel_positions

# ===================== HDF5Writer =====================
class HDF5Writer:
    """
    HDF5 file writer.
    
    Responsible for writing processed EEG data to HDF5 format files, including:
    - Subject-level attributes (SubjectAttrs)
    - Trial-level attributes (TrialAttrs)
    - Segment-level data and attributes (SegmentAttrs)
    """
    def __init__(self, file_path: str, subject_attrs: SubjectAttrs):
        """
        Initialize HDF5 writer.
        
        Args:
            file_path: Output HDF5 file path
            subject_attrs: Subject attributes object
        """
        self.file_path = file_path
        self.hf = h5py.File(file_path, 'w')
        self.trial_groups = {}
        self.chn_names = subject_attrs.chn_name
        self.chn_count = len(subject_attrs.chn_name)
        self.chn_pos = subject_attrs.chn_pos  # Channel positions (may be None)
        
        self._write_subject_attrs(subject_attrs)

    def _write_subject_attrs(self, subject_attrs: SubjectAttrs):
        """
        Write subject-level attributes to HDF5 file.
        
        Args:
            subject_attrs: Subject attributes object
        """
        subject_dict = subject_attrs.to_dict()
        for key, value in subject_dict.items():
            if key in ["chn_name", "category_list"] and isinstance(value, list) and all(isinstance(x, str) for x in value):
                # Convert string list to HDF5 string array
                dt = h5py.string_dtype(encoding='utf-8')
                self.hf.attrs[key] = np.array(value, dtype=dt)
            elif key == "chn_pos":
                # Handle channel positions: compatible with string array and None value
                if isinstance(value, np.ndarray) and value.dtype.kind == 'U':
                    self.hf.attrs[key] = value
                else:
                    self.hf.attrs[key] = str(value) if value is not None else "None"
            else:
                self.hf.attrs[key] = value

    def add_trial(self, trial_attrs: TrialAttrs) -> str:
        """
        Add trial to HDF5 file.
        
        Args:
            trial_attrs: Trial attributes object
        
        Returns:
            str: Trial name (e.g., 'trial_0')
        """
        trial_dict = trial_attrs.to_dict()
        trial_id = trial_dict['trial_id']
        trial_name = f'trial_{trial_id}'
        
        if trial_name not in self.hf:
            trial_group = self.hf.create_group(trial_name)
            for key, value in trial_dict.items():
                trial_group.attrs[key] = value
            self.trial_groups[trial_id] = trial_group
        
        return trial_name

    def add_segment(self, trial_name: str, segment_attrs: SegmentAttrs, segment_data: np.ndarray):
        """
        Add segment to HDF5 file.
        
        Note: The label field should be a numeric array (numpy.ndarray), format: np.array([label_index])
        where label_index is 0 (perceive) or 1 (imagine).
        
        Args:
            trial_name: Trial name (e.g., 'trial_0')
            segment_attrs: Segment attributes object
            segment_data: Segment data (shape: n_channels x n_samples), unit: µV
        """
        segment_dict = segment_attrs.to_dict()
        segment_id = segment_dict['segment_id']
        trial_group = self.hf[trial_name]
        
        # Create segment group
        segment_group_name = f'segment{segment_id}'
        if segment_group_name not in trial_group:
            segment_group = trial_group.create_group(segment_group_name)
        else:
            segment_group = trial_group[segment_group_name]
        
        # Write EEG dataset
        seg_data_numeric = segment_data.astype(np.float32)
        eeg_dataset = segment_group.create_dataset(
            'eeg',
            data=seg_data_numeric,
            dtype=np.float32,
            compression='gzip'
        )
        
        # Write segment metadata
        for key, value in segment_dict.items():
            if key == "label":
                # Label should be numeric array, convert to integer array
                if isinstance(value, np.ndarray):
                    eeg_dataset.attrs[key] = value.astype(np.int64)
                else:
                    # Compatible with old format: if single value, convert to array
                    eeg_dataset.attrs[key] = np.array([int(value)], dtype=np.int64)
            elif isinstance(value, np.ndarray):
                eeg_dataset.attrs[key] = value.astype(np.float32)
            elif isinstance(value, (int, float, str)):
                eeg_dataset.attrs[key] = value
            else:
                eeg_dataset.attrs[key] = str(value)
        
        # Write channel information
        eeg_dataset.attrs['channel_count'] = self.chn_count
        dt = h5py.string_dtype(encoding='utf-8')
        eeg_dataset.attrs['channel_names'] = np.array(self.chn_names, dtype=dt)
        
        # Write channel positions (if available)
        if self.chn_pos is not None and isinstance(self.chn_pos, list) and len(self.chn_pos) > 0:
            chn_pos_pure_str = [str(item).strip() if item is not None else " " for item in self.chn_pos]
            eeg_dataset.attrs['channel_positions'] = np.array(chn_pos_pure_str, dtype=dt)
        else:
            eeg_dataset.attrs['channel_positions'] = "None"

    def close(self):
        self.hf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# ===================== Multi-Subject Dataset Builder =====================
class MultiSubjectDatasetBuilder:
    """
    Multi-subject dataset builder.
    
    Responsible for processing all subjects in the PerceiveImagine EEG dataset, including:
    - Reading EEGLAB .set files
    - Loading event and channel information
    - Preprocessing (filtering, resampling)
    - Extracting trials and segmenting into windows
    - Label mapping (original value -> class index)
    - Writing to HDF5 format
    - Generating dataset info JSON
    """
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = DEFAULT_TARGET_SFREQ,
        window_sec: float = DEFAULT_WINDOW_SEC,
        stride_sec: float = DEFAULT_STRIDE_SEC,
        filter_low: float = DEFAULT_FILTER_LOW,
        filter_high: float = DEFAULT_FILTER_HIGH,
        filter_notch: float = DEFAULT_FILTER_NOTCH,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV
    ):
        """
        Initialize multi-subject dataset builder.
        
        Args:
            raw_data_dir: Raw data root directory (contains all sub-* folders)
            output_dir: Output directory
            target_sfreq: Target sampling rate (Hz)
            window_sec: Window length (seconds)
            stride_sec: Window stride (seconds)
            filter_low: Low cutoff frequency for bandpass filter (Hz)
            filter_high: High cutoff frequency for bandpass filter (Hz)
            filter_notch: Notch filter frequency (Hz)
            max_amplitude_uv: Amplitude validation threshold (µV)
        """
        self.raw_data_root = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        
        # Statistics tracking
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0
        
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.trial_reserve_samples = int(TRIAL_RESERVE_SEC * target_sfreq)
        
        # Channel information (will be set during first subject processing)
        self.channel_names = None
        self.channel_positions = None  # Channel positions (if available)
        
        self.subject_set_files = self._find_all_subject_set_files()
        if not self.subject_set_files:
            raise FileNotFoundError("No .set files found in sub-*/eeg directories")
        print(f"Found {len(self.subject_set_files)} subject dataset files, starting batch processing")

    def _find_all_subject_set_files(self) -> dict:
        """
        Find all subject .set files.
        
        Searches for files matching the naming pattern: sub-*_task-PerceiveImagine_eeg.set
        
        Returns:
            dict: {subject_id: set_file_path} dictionary
        """
        subject_set_files = {}
        subject_dirs = list(self.raw_data_root.glob("sub-*"))
        
        for sub_dir in subject_dirs:
            if not sub_dir.is_dir():
                continue
            
            eeg_dir = sub_dir / "eeg"
            if not eeg_dir.exists():
                print(f"Warning: {sub_dir} has no eeg folder, skipping")
                continue
            
            set_files = list(eeg_dir.glob("sub-*_task-PerceiveImagine_eeg.set"))
            if not set_files:
                print(f"Warning: {eeg_dir} has no matching .set files, skipping")
                continue
            
            subject_id = sub_dir.name.split("-")[-1]
            try:
                subject_id_int = int(subject_id)
            except ValueError:
                subject_id_int = subject_id
            
            subject_set_files[subject_id_int] = set_files[0]
        
        return subject_set_files

    def _read_raw(self, set_file_path: Path) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame, dict]:
        """
        Read raw EEG data file.
        
        This function will:
        1. Validate file existence and validity
        2. Load EEGLAB .set file
        3. Load auxiliary files (channels.tsv, events.tsv, JSON)
        4. Process channel names and positions
        5. Process event label mapping
        6. Convert data units and rebuild MNE Raw object
        
        Args:
            set_file_path: Path to .set file
        
        Returns:
            tuple: (raw_processed, channels_df, events_df, eeg_metadata)
                - raw_processed: Processed MNE Raw object
                - channels_df: Channel information DataFrame
                - events_df: Event information DataFrame (contains label_index and label_name columns)
                - eeg_metadata: JSON metadata dictionary
        
        Raises:
            OSError: If file does not exist or is corrupted
            ValueError: If file format is incorrect
        """
        # File validity check
        if not set_file_path.exists():
            raise OSError(f"File does not exist: {str(set_file_path)}")
        
        # Check file size (minimum 1KB, can be adjusted as needed)
        file_size = set_file_path.stat().st_size
        if file_size < 1024:  # Less than 1KB is considered abnormal
            raise OSError(f"Abnormal file size ({file_size} bytes), may be corrupted: {str(set_file_path)}")
        
        if set_file_path.suffix.lower() != '.set':
            raise ValueError(f"Unsupported file format: {set_file_path.suffix}")
        
        # 1. Load .set file
        # Note: preload=True is necessary for preprocessing, but we'll release it after extracting data
        raw = mne.io.read_raw_eeglab(str(set_file_path), preload=True, verbose=False)
       
        # 2. Load auxiliary files
        channels_df, events_df, eeg_metadata, channel_names, channel_positions = load_sub01_aux_files(set_file_path)
        
        # 3. Extract channel position information (if available)
        try:
            if raw.get_montage() is not None:
                ch_pos = raw.get_montage().get_positions()['ch_pos']
                # Extract channel positions and convert to string list
                self.channel_positions = [
                    str(ch_pos.get(name, " ")).strip() if ch_pos.get(name) is not None else " "
                    for name in channel_names
                ]
                # If all positions are empty strings, set to None
                if len(self.channel_positions) == 0 or all(item == " " for item in self.channel_positions):
                    self.channel_positions = None
            else:
                # No montage information, set to None
                self.channel_positions = None
        except Exception:
            # Extraction failed, set to None
            self.channel_positions = None
        
        # 4. Save channel names
        self.channel_names = channel_names
        
        # 5. Channel filtering and rebuild Raw object
        eeg_ch_mask = [ch in self.channel_names for ch in raw.ch_names]
        # Store sfreq before processing
        sfreq = raw.info['sfreq']
        eeg_data = raw.get_data()[eeg_ch_mask, :]
        data_volts, _ = detect_unit_and_convert_to_volts(eeg_data)
        
        # Release original raw data to free memory (we have eeg_data and sfreq now)
        del raw
        import gc
        gc.collect()
        
        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=sfreq,
            ch_types=['eeg'] * len(self.channel_names)
        )
        raw_processed = mne.io.RawArray(data_volts, info, verbose=False)
        
        # 6. Return processed data
        return raw_processed, channels_df, events_df, eeg_metadata

    def _preprocess(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Preprocess raw data.
        
        Preprocessing steps:
        1. Reference electrode setup (use reference channels if available, otherwise use average reference)
        2. Notch filter (remove power line interference)
        3. Bandpass filter (keep valid frequency band)
        4. Resampling (if needed)
        
        Args:
            raw: MNE Raw object
        
        Returns:
            mne.io.Raw: Preprocessed Raw object
        """
        ref_chs = [ch for ch in REFERENCE_CHANNELS if ch in raw.ch_names]
        if ref_chs:
            raw.set_eeg_reference(ref_channels=ref_chs, verbose=False)
            raw.drop_channels(ref_chs)
        else:
            raw.set_eeg_reference('average', verbose=False)
        
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        
        return raw

    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """
        Validate trial data amplitude is within allowed range.
        
        Args:
            trial_data: Trial data (shape: n_channels x n_samples), unit: µV
        
        Returns:
            bool: True if amplitude is within threshold, False otherwise
        """
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _process_single_subject(self, subject_id: int, set_file_path: Path) -> str:
        """
        Process data for a single subject.
        
        Processing pipeline:
        1. Read raw data and event information
        2. Preprocess (filtering, resampling)
        3. Extract trials (based on event timestamps)
        4. Validate trial amplitude
        5. Segment trials into windows
        6. Write to HDF5 file
        
        Args:
            subject_id: Subject ID
            set_file_path: Path to .set file
        
        Returns:
            str: Output HDF5 file path, empty string if processing failed
        """
        # Reset instance attributes (avoid data contamination between subjects)
        self.channel_names = None
        self.channel_positions = None
        
        print(f"\n===== Processing subject {subject_id} =====")
        print(f"Reading {set_file_path}")
        
        # Read data
        try:
            raw, channels_df, events_df, eeg_metadata = self._read_raw(set_file_path)
        except (OSError, Exception) as e:
            print(f"Error: Failed to read file for subject {subject_id}, skipping. Error: {str(e)}")
            return ""
        
        try:
            # Data preprocessing
            raw = self._preprocess(raw)
            
            # Data conversion: MNE uses V internally, convert to µV for storage
            # Use get_data() only once and keep reference for trial extraction
            data_volts = raw.get_data()
            data_uv = data_volts * 1e6
            print(f"Subject {subject_id} data max amplitude (µV): {np.abs(data_uv).max():.1f}")
            
            # Release raw object to free memory (we have data_uv now)
            del raw
            import gc
            gc.collect()

            # Build SubjectAttrs (before processing trials)
            subject_attrs = SubjectAttrs(
                subject_id=subject_id,
                dataset_name=PERCEIVE_IMAGINE_INFO.dataset_name,
                task_type=PERCEIVE_IMAGINE_INFO.task_type,
                downstream_task_type=PERCEIVE_IMAGINE_INFO.downstream_task_type,
                rsFreq=self.target_sfreq,
                chn_name=self.channel_names,
                num_labels=PERCEIVE_IMAGINE_INFO.num_labels,
                category_list=PERCEIVE_IMAGINE_INFO.category_list,
                chn_pos=self.channel_positions,
                chn_ori=None,
                chn_type="EEG",
                montage=PERCEIVE_IMAGINE_INFO.montage,
            )

            # Statistics tracking
            total_trials = 0
            valid_trials = 0
            rejected_trials = 0
            
            # Write to HDF5 (streaming: process and write one trial at a time)
            output_h5_path = self.output_dir / f"sub_{subject_id}.h5"
            with HDF5Writer(str(output_h5_path), subject_attrs) as writer:
                # Iterate through all events to extract and write trials one by one
                for trial_idx, (_, event_row) in enumerate(events_df.iterrows(), 1):
                    onset_time = event_row['onset']
                    label_index = event_row['label_index']  # Numeric label index (0 or 1)
                    label_name = event_row['label_name']  # Label name ("perceive" or "imagine")
                    
                    start_sample = int(onset_time * self.target_sfreq)
                    end_sample = start_sample + self.trial_reserve_samples
                    
                    if end_sample > data_uv.shape[1]:
                        end_sample = data_uv.shape[1]
                        print(f"Warning: Subject {subject_id} trial {trial_idx} exceeds data range, truncated")
                    
                    # Extract trial data (view, not copy, to save memory)
                    trial_data_uv = data_uv[:, start_sample:end_sample].copy()  # Copy needed for validation
                    
                    total_trials += 1
                    if not self._validate_trial(trial_data_uv):
                        rejected_trials += 1
                        del trial_data_uv  # Release memory immediately
                        continue
                    valid_trials += 1
                    
                    # Write trial immediately (don't store in memory)
                    trial_attrs = TrialAttrs(
                        trial_id=trial_idx,
                        session_id=1,
                    )
                    trial_name = writer.add_trial(trial_attrs)
                    
                    n_samples = trial_data_uv.shape[1]
                    for i_slice, start in enumerate(
                        range(0, n_samples - self.window_samples + 1, self.stride_samples)
                    ):
                        end = start + self.window_samples
                        slice_data = trial_data_uv[:, start:end]  # View, not copy
                        
                        # Use numeric label array (conforms to SegmentAttrs specification)
                        label_array = np.array([label_index], dtype=np.int64)
                        
                        segment_attrs = SegmentAttrs(
                            segment_id=i_slice,
                            start_time=onset_time + start / self.target_sfreq,
                            end_time=onset_time + end / self.target_sfreq,
                            time_length=self.window_sec,
                            label=label_array,  # Numeric label array
                        )
                        writer.add_segment(trial_name, segment_attrs, slice_data)
                    
                    # Release trial data immediately after writing
                    del trial_data_uv
                    if trial_idx % 10 == 0:  # Periodic garbage collection every 10 trials
                        gc.collect()
            
            # Release main data array after all trials processed
            del data_uv, data_volts
            gc.collect()
            
            if valid_trials == 0:
                print(f"Warning: Subject {subject_id} has no valid trials, skipping")
                return ""

            # Update statistics
            self.total_trials += total_trials
            self.valid_trials += valid_trials
            self.rejected_trials += rejected_trials
            
            # Report statistics
            print(f"\nSubject {subject_id} statistics:")
            print(f"  Total trials: {total_trials}")
            print(f"  Valid trials: {valid_trials}")
            print(f"  Rejected trials: {rejected_trials}")
            if total_trials > 0:
                print(f"  Rejection rate: {rejected_trials / total_trials * 100:.1f}%")
            
            print(f"Subject {subject_id} processing completed, saved to {output_h5_path}")
            
            return str(output_h5_path)
        
        except Exception as e:
            print(f"Error: Unknown exception occurred while processing subject {subject_id}, skipping. Error: {str(e)}")
            return ""

    def _save_dataset_info(self, stats: dict):
        """
        Save dataset information to dataset_info.json.
        
        This function is called after all subjects are processed to generate a unified dataset info file.
        
        Args:
            stats: Statistics dictionary containing processing statistics
        """
        # Process channel position information
        chn_pos_save = self.channel_positions if self.channel_positions is not None else "None"
        if isinstance(chn_pos_save, list):
            chn_pos_save = [str(item).strip() if item is not None else " " for item in chn_pos_save]

        dataset_info = {
            "dataset": {
                "name": PERCEIVE_IMAGINE_INFO.dataset_name,
                "description": "PerceiveImagine EEG Dataset - Perceive vs Imagine binary classification",
                "task_type": PERCEIVE_IMAGINE_INFO.task_type,
                "downstream_task": PERCEIVE_IMAGINE_INFO.downstream_task_type,
                "num_labels": PERCEIVE_IMAGINE_INFO.num_labels,
                "category_list": PERCEIVE_IMAGINE_INFO.category_list,
                "label_mapping": {
                    "original_value -> (class_index, class_name)": {
                        "1": "(0, 'perceive')",
                        "2": "(1, 'imagine')",
                        "255": "(None, 'n/a') - invalid label, filtered"
                    }
                },
                "channels": self.channel_names if self.channel_names else [],
                "montage": PERCEIVE_IMAGINE_INFO.montage,
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "trial_reserve_sec": TRIAL_RESERVE_SEC,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }

        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"Saved dataset info to {json_path}")

    def build_all_subjects(self) -> list[str]:
        """
        Process all subjects in batch.
        
        Returns:
            list[str]: List of all successfully processed output file paths
        """
        # Reset statistics
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0
        
        all_output_paths = []
        failed_subjects = []
        
        # Sort subjects by ID in ascending order; ID may be int or str, use robust sorting
        def _sort_key(item):
            subj_id, _ = item
            # Prefer integer sorting, then string sorting
            try:
                return int(subj_id)
            except (TypeError, ValueError):
                return str(subj_id)

        sorted_subjects = sorted(self.subject_set_files.items(), key=_sort_key)
        
        # Iterate through all available subjects
        for subject_id, set_file_path in sorted_subjects:
            output_path = self._process_single_subject(subject_id, set_file_path)
            if output_path:
                all_output_paths.append(output_path)
            else:
                failed_subjects.append(subject_id)
        
        # Generate summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(sorted_subjects)}")
        print(f"Successful: {len(all_output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal trials: {self.total_trials}")
        print(f"Valid trials: {self.valid_trials}")
        print(f"Rejected trials: {self.rejected_trials}")
        if self.total_trials > 0:
            print(f"Rejection rate: {self.rejected_trials / self.total_trials * 100:.1f}%")
        print("=" * 50)
        
        # Save dataset information
        stats = {
            "total_subjects": len(sorted_subjects),
            "successful_subjects": len(all_output_paths),
            "failed_subjects": failed_subjects,
            "total_trials": self.total_trials,
            "valid_trials": self.valid_trials,
            "rejected_trials": self.rejected_trials,
            "rejection_rate": f"{self.rejected_trials / self.total_trials * 100:.1f}%" if self.total_trials > 0 else "0%",
        }
        self._save_dataset_info(stats)
        
        print(f"\nAll subjects processing completed, results saved to {self.output_dir}")
        return all_output_paths

# ===================== Convenience Functions and CLI Entry =====================
def build_multi_subject_dataset(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    **kwargs,
) -> list[str]:
    """
    Convenience function: Build multi-subject dataset.
    
    Args:
        raw_data_dir: Raw data root directory (contains all sub-* folders)
        output_dir: Output directory
        **kwargs: Additional arguments passed to MultiSubjectDatasetBuilder
    
    Returns:
        list[str]: List of all successfully processed output file paths
    """
    builder = MultiSubjectDatasetBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all_subjects()

def main():
    """
    Command-line entry point.
    """
    parser = argparse.ArgumentParser(
        description="PerceiveImagine EEG Dataset Preprocessing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m benchmark_dataloader.datasets.perceiveimage /path/to/raw/data --output_dir ./output
  
Label mapping:
  - Original label value 1 -> class index 0 (perceive)
  - Original label value 2 -> class index 1 (imagine)
  - Original label value 255 -> invalid label, will be filtered
        """
    )
    parser.add_argument("raw_data_dir", help="Raw data root directory (contains all sub-* folders)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory (default: ./hdf5)")
    parser.add_argument("--target_sfreq", type=float, default=DEFAULT_TARGET_SFREQ, 
                        help=f"Target sampling rate (default: {DEFAULT_TARGET_SFREQ} Hz)")
    parser.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC, 
                        help=f"Segment window length (seconds, default: {DEFAULT_WINDOW_SEC})")
    parser.add_argument("--stride_sec", type=float, default=DEFAULT_STRIDE_SEC, 
                        help=f"Segment window stride (seconds, default: {DEFAULT_STRIDE_SEC})")
    parser.add_argument("--filter_low", type=float, default=DEFAULT_FILTER_LOW,
                        help=f"Low cutoff frequency for bandpass filter (Hz, default: {DEFAULT_FILTER_LOW})")
    parser.add_argument("--filter_high", type=float, default=DEFAULT_FILTER_HIGH,
                        help=f"High cutoff frequency for bandpass filter (Hz, default: {DEFAULT_FILTER_HIGH})")
    parser.add_argument("--filter_notch", type=float, default=DEFAULT_FILTER_NOTCH,
                        help=f"Notch filter frequency (Hz, default: {DEFAULT_FILTER_NOTCH})")
    parser.add_argument("--max_amplitude_uv", type=float, default=DEFAULT_MAX_AMPLITUDE_UV,
                        help=f"Amplitude validation threshold (µV, default: {DEFAULT_MAX_AMPLITUDE_UV})")
    args = parser.parse_args()

    build_multi_subject_dataset(
        args.raw_data_dir,
        args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv
    )

if __name__ == '__main__':
    main()