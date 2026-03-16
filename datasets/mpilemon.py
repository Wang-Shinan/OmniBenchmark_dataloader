"""
MPI-LEMON Resting-state EEG Dataset Builder.

Dataset: MPILMBB_LEMON
- BrainVision format (.vhdr/.eeg/.vmrk)
- Resting-state EEG (RSEEG) per subject
- Rich behavioural/clinical metadata (age, gender, cognitive & personality scores)

This builder:
- Reads raw resting EEG from `EEG_Raw_BIDS_ID/sub-*/RSEEG/sub-xxxxx.vhdr`
- Bandpass: 0.1–75 Hz, optional 50 Hz notch
- Resample to 200 Hz
- Segment into 1 s non-overlapping windows
- Writes multiple HDF5 variants with different labels (gender, age_group, etc.)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..utils import ElectrodeSet
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from utils import ElectrodeSet  # type: ignore
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs  # type: ignore
    from hdf5_io import HDF5Writer  # type: ignore
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType  # type: ignore


DEFAULT_MAX_AMPLITUDE_UV = 800.0

# Standard 61-channel configuration (most common in MPI-LEMON)
# Based on the most frequent 61-channel setup found in the dataset
STANDARD_61_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'AFz', 
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 
    'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 
    'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 
    'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'
]

# Base dataset info (resting-state)
MPI_LEMON_BASE_INFO = DatasetInfo(
    dataset_name="MPI_LEMON_Resting",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,  # will be overridden per label variant
    category_list=["1:older than 40","0:younger than 40"],
    sampling_rate=200.0,
    montage="10_10",
    channels=STANDARD_61_CHANNELS,  # Standard 61-channel configuration
)


def clean_channel_name(name: str) -> str:
    """Clean and normalize channel name."""
    name = str(name).upper().strip()
    # Remove common prefixes/suffixes
    name = name.replace("EEG ", "").replace("-REF", "").replace("REF", "")
    return name


def standardize_channel_name(name: str, electrode_set: ElectrodeSet) -> str:
    """Standardize channel name using ElectrodeSet."""
    clean = clean_channel_name(name)
    return electrode_set.standardize_name(clean)


def map_channels_to_standard_61(
    data: np.ndarray,
    source_channels: List[str],
    target_channels: List[str] = STANDARD_61_CHANNELS,
    electrode_set: Optional[ElectrodeSet] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Map data from source channels to standard 61-channel configuration.
    
    Args:
        data: Input data array (shape: n_source_channels x n_samples)
        source_channels: List of source channel names
        target_channels: List of target channel names (default: STANDARD_61_CHANNELS)
        electrode_set: ElectrodeSet instance for name standardization
    
    Returns:
        Tuple of (mapped_data, target_channels)
        - mapped_data: Output data array (shape: n_target_channels x n_samples)
        - target_channels: List of target channel names
    """
    if electrode_set is None:
        electrode_set = ElectrodeSet()
    
    n_source, n_samples = data.shape
    n_target = len(target_channels)
    
    # Create source channel map (standardized names -> index)
    source_ch_map = {}
    for idx, ch in enumerate(source_channels):
        std_name = standardize_channel_name(ch, electrode_set)
        source_ch_map[std_name] = idx
    
    # Initialize output array
    mapped_data = np.zeros((n_target, n_samples), dtype=data.dtype)
    
    # Map each target channel
    for target_idx, target_ch in enumerate(target_channels):
        target_std = standardize_channel_name(target_ch, electrode_set)
        
        # Case 1: Direct match
        if target_std in source_ch_map:
            source_idx = source_ch_map[target_std]
            mapped_data[target_idx, :] = data[source_idx, :]
            continue
        
        # Case 2: Try to find similar channel names (for aliases)
        # Check if any source channel matches after standardization
        found_match = False
        for src_ch in source_channels:
            src_std = standardize_channel_name(src_ch, electrode_set)
            if src_std == target_std:
                source_idx = source_channels.index(src_ch)
                mapped_data[target_idx, :] = data[source_idx, :]
                found_match = True
                break
        
        if found_match:
            continue
        
        # Case 3: Interpolation from nearest neighbors
        # Find channels with similar names (e.g., FC3 -> FC1, FC2, FC5, FC6)
        # Strategy: match by prefix (e.g., FC) and find nearby channels
        neighbors = []
        target_prefix = target_std[:2] if len(target_std) >= 2 else target_std
        
        # First, try to find channels with same prefix
        for src_ch in source_channels:
            src_std = standardize_channel_name(src_ch, electrode_set)
            if src_std.startswith(target_prefix):
                neighbors.append(source_channels.index(src_ch))
                if len(neighbors) >= 4:  # Use up to 4 neighbors
                    break
        
        # If not enough neighbors, try channels with similar positions
        if len(neighbors) < 2:
            # Try to find channels that are spatially close
            # For example, if target is FC3, try FC1, FC2, FC5, FC6, F3, C3, etc.
            for src_ch in source_channels:
                src_std = standardize_channel_name(src_ch, electrode_set)
                # Check if it's a nearby channel (shares one letter position)
                if (target_prefix[0] in src_std and target_prefix[1] in src_std) or \
                   (len(target_std) > 2 and target_std[2] in src_std):
                    if source_channels.index(src_ch) not in neighbors:
                        neighbors.append(source_channels.index(src_ch))
                        if len(neighbors) >= 4:
                            break
        
        if neighbors:
            # Average interpolation
            mapped_data[target_idx, :] = np.mean(data[neighbors, :], axis=0)
        else:
            # No neighbors found, fill with zeros
            # This should rarely happen if the dataset is reasonably complete
            mapped_data[target_idx, :] = 0.0
    
    return mapped_data, target_channels


def detect_unit_and_convert_to_volts(data: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Robustly detect unit (V/mV/µV) and convert to Volts.
    """
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))

    mad = np.median(abs_data)
    if mad > 0:
        mad_based = 3 * mad
        max_amp = max(max_amp, mad_based)

    if max_amp > 1e-2:
        return data / 1e6, "µV"
    elif max_amp > 1e-5:
        return data / 1e3, "mV"
    else:
        return data, "V"


@dataclass
class LabelConfig:
    """Configuration for one label variant (e.g., gender, age group)."""

    name: str
    category_list: List[str]
    # mapping from raw metadata (e.g., "1", "2", "65-70") to label index
    mapping: Dict[str, int]


class MPILemonBuilder:
    """
    Builder for MPI-LEMON resting EEG with multiple label variants.
    """

    def __init__(
        self,
        raw_root: str,
        beh_root: str,
        output_root: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        Args:
            raw_root: Path to `EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID`
            beh_root: Path to `Behavioural_Data_MPILMBB_LEMON`
            output_root: Base output directory
        """
        self.raw_root = Path(raw_root)
        self.beh_root = Path(beh_root)
        self.output_root = Path(output_root)

        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)

        # metadata tables
        self._meta_age_gender: Optional[pd.DataFrame] = None
        self._neo_ffi: Optional[pd.DataFrame] = None  # Big Five personality data
        
        # Extraversion median for binary classification
        self._extraversion_median: Optional[float] = None

        # label configurations (can be extended later)
        self.label_configs: Dict[str, LabelConfig] = self._init_label_configs()
        
        # Store standard channels (always use 61-channel standard)
        self._dataset_channels: Optional[List[str]] = STANDARD_61_CHANNELS
        
        # ElectrodeSet for channel standardization
        self.electrode_set = ElectrodeSet()
        
        # Track validation statistics per label variant
        self._stats_per_label: Dict[str, Dict[str, int]] = {
            name: {
                "total_segments": 0,
                "valid_segments": 0,
                "rejected_segments": 0,
            }
            for name in self.label_configs
        }

    # ------------------------------------------------------------------
    # Metadata & labels
    # ------------------------------------------------------------------
    def _load_age_gender_meta(self) -> pd.DataFrame:
        """
        Load basic age/gender meta file:
        `META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv`
        """
        if self._meta_age_gender is None:
            meta_path = (
                self.beh_root
                / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
            )
            if not meta_path.exists():
                raise FileNotFoundError(f"Meta file not found: {meta_path}")
            df = pd.read_csv(meta_path)
            # Ensure ID is string, consistent with BIDS IDs (e.g., "sub-032301")
            if "ID" not in df.columns:
                raise ValueError("Meta file missing 'ID' column")
            df["ID"] = df["ID"].astype(str).str.strip()
            self._meta_age_gender = df
        return self._meta_age_gender

    def _load_neo_ffi(self) -> pd.DataFrame:
        """Load NEO-FFI (Big Five personality) data."""
        if self._neo_ffi is None:
            neo_path = (
                self.beh_root
                / "Emotion_and_Personality_Test_Battery_LEMON"
                / "NEO_FFI.csv"
            )
            if not neo_path.exists():
                raise FileNotFoundError(f"NEO-FFI file not found: {neo_path}")
            df = pd.read_csv(neo_path)
            # First column is subject ID (Unnamed: 0)
            df.rename(columns={df.columns[0]: "ID"}, inplace=True)
            df["ID"] = df["ID"].astype(str).str.strip()
            self._neo_ffi = df
        return self._neo_ffi

    def _init_label_configs(self) -> Dict[str, LabelConfig]:
        """
        Define label variants to export.

        Currently:
        - gender: 2 classes (female, male)
        - age_group: multiple age ranges as in Participants file
        - extraversion: 2 classes (low, high) based on median split
        """
        # Gender: 1=female, 2=male
        gender_categories = ["female", "male"]
        gender_mapping = {"1": 0, "2": 1}

        # Age group: define two coarse groups for MPI_LEMON_Resting_age_group
        #   - young (<= 40)
        #   - old   (>= 55)
        # 注意：Participants_MPILMBB_LEMON.csv 位于 MPI-LEMON 根目录，
        # 而不是 Behavioural_Data_MPILMBB_LEMON 目录下，所以用 beh_root 的父目录
        participants_path = (
            self.beh_root.parent / "Participants_MPILMBB_LEMON.csv"
        )
        if not participants_path.exists():
            raise FileNotFoundError(
                f"Participants file not found: {participants_path}"
            )
        part_df = pd.read_csv(participants_path)
        if "Age" not in part_df.columns or "ID" not in part_df.columns:
            raise ValueError("Participants file missing 'ID' or 'Age' columns")

        # Define 2-way age groups for MPI_LEMON_Resting_age_group:
        # - 'young_<=40' : all subjects whose age interval is fully <= 40
        # - 'old_>=55'   : all subjects whose age interval is fully >= 55
        # Any subjects with age ranges overlapping (40, 55) are left unlabeled
        # for this task (labels['age_group'] will be None and they are skipped).
        age_categories = ["young_<=40", "old_>=55"]
        age_mapping = {
            "young_<=40": 0,
            "old_>=55": 1,
        }

        # Extraversion: binary classification based on median split
        # Load NEO-FFI data to compute median
        neo_df = self._load_neo_ffi()
        extraversion_values = neo_df["NEOFFI_Extraversion"].dropna()
        if len(extraversion_values) == 0:
            raise ValueError("No Extraversion data found in NEO-FFI file")
        
        self._extraversion_median = float(extraversion_values.median())
        print(f"Extraversion median: {self._extraversion_median:.4f}")
        
        extraversion_categories = ["low_extraversion", "high_extraversion"]
        # Mapping is done dynamically in _get_labels_for_subject based on median comparison
        # Use empty dict as placeholder (actual mapping done at runtime)
        extraversion_mapping = {}

        return {
            "gender": LabelConfig(
                name="gender",
                category_list=gender_categories,
                mapping=gender_mapping,
            ),
            "age_group": LabelConfig(
                name="age_group",
                category_list=age_categories,
                mapping=age_mapping,
            ),
            "extraversion": LabelConfig(
                name="extraversion",
                category_list=extraversion_categories,
                mapping=extraversion_mapping,  # Special mapping with median
            ),
        }

    def interval_to_bin(self, interval: str) -> Optional[str]:
        """Map an age interval (e.g. '20-25') to a coarse age bin.

        We define two bins for MPI_LEMON_Resting_age_group:
        - 'young_<=40' for subjects whose entire age interval is <= 40
        - 'old_>=55'   for subjects whose entire age interval is >= 55

        Any interval that overlaps the middle range (40, 55) returns None,
        meaning this subject will not receive an age_group label.
        """
        interval = str(interval).strip()
        if not interval:
            return None

        # Handle simple single-value ages as well as ranges like '20-25'
        try:
            if "-" in interval:
                parts = interval.split("-")
                lower = float(parts[0])
                upper = float(parts[1])
            else:
                lower = upper = float(interval)
        except ValueError:
            return None

        if upper <= 40.0:
            return "young_<=40"
        if lower >= 55.0:
            return "old_>=55"
        return None

    def _get_labels_for_subject(
        self, bids_id: str
    ) -> Dict[str, Optional[int]]:
        """
        Get label indices for a subject for all label variants.

        Args:
            bids_id: e.g., "sub-032301"
        """
        labels: Dict[str, Optional[int]] = {k: None for k in self.label_configs}

        # Gender & basic age categories from Participants file
        participants_path = (
            self.beh_root.parent / "Participants_MPILMBB_LEMON.csv"
        )
        part_df = pd.read_csv(participants_path)
        part_df["ID"] = part_df["ID"].astype(str).str.strip()
        row = part_df[part_df["ID"] == bids_id]
        if not row.empty:
            row = row.iloc[0]
            # gender: "Gender_ 1=female_2=male"
            gender_col = [c for c in row.index if "Gender" in c][0]
            gender_raw = str(row[gender_col]).strip()
            cfg_gender = self.label_configs["gender"]
            labels["gender"] = cfg_gender.mapping.get(gender_raw, None)

            # age_group
            age_raw = str(row["Age"]).strip()
            age_bin = self.interval_to_bin(age_raw)
            cfg_age = self.label_configs["age_group"]
            labels["age_group"] = cfg_age.mapping.get(age_bin, None)

        # Extraversion: binary classification based on median split
        neo_df = self._load_neo_ffi()
        neo_row = neo_df[neo_df["ID"] == bids_id]
        if not neo_row.empty:
            extraversion_value = neo_row["NEOFFI_Extraversion"].iloc[0]
            if pd.notna(extraversion_value):
                # 0 = low_extraversion (<= median), 1 = high_extraversion (> median)
                if extraversion_value <= self._extraversion_median:
                    labels["extraversion"] = 0
                else:
                    labels["extraversion"] = 1

        return labels

    # ------------------------------------------------------------------
    # File discovery & reading
    # ------------------------------------------------------------------
    def get_subject_ids(self) -> List[str]:
        """Return list of BIDS IDs with raw RSEEG data."""
        subjects = []
        for sub_dir in sorted(self.raw_root.glob("sub-*")):
            rseeg_dir = sub_dir / "RSEEG"
            if not rseeg_dir.exists():
                continue
            vhdr_files = list(rseeg_dir.glob("*.vhdr"))
            if vhdr_files:
                subjects.append(sub_dir.name)  # e.g., "sub-032301"
        return subjects

    def _find_raw_file(self, bids_id: str) -> Path:
        sub_dir = self.raw_root / bids_id / "RSEEG"
        vhdr_files = list(sub_dir.glob("*.vhdr"))
        if not vhdr_files:
            raise FileNotFoundError(f"No .vhdr file found for {bids_id} in {sub_dir}")
        return vhdr_files[0]

    def _read_preprocessed_eeglab(self, bids_id: str) -> mne.io.BaseRaw:
        """
        Fallback reader: use preprocessed EEGLAB .set (eyes-closed) if BrainVision
        header points to missing files for some subjects.
        """
        preproc_root = (
            self.raw_root.parent
            / "EEG_Preprocessed_BIDS_ID"
            / "EEG_Preprocessed"
            / bids_id
        )
        if not preproc_root.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preproc_root}")

        # Prefer eyes-closed (EC) resting
        ec_files = list(preproc_root.glob("*_EC.set"))
        eo_files = list(preproc_root.glob("*_EO.set"))

        set_file: Optional[Path]
        if ec_files:
            set_file = ec_files[0]
        elif eo_files:
            set_file = eo_files[0]
        else:
            raise FileNotFoundError(
                f"No *_EC.set or *_EO.set file found in {preproc_root}"
            )

        print(f"  [fallback] using preprocessed EEGLAB file: {set_file.name}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)

        # Drop non-EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
        raw.pick(eeg_picks)

        # Units: EEGLAB data often in µV; detect & convert to V
        data = raw.get_data()
        data_v, _ = detect_unit_and_convert_to_volts(data)
        raw._data = data_v

        # Apply same preprocessing chain (filter + resample)
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            verbose=False,
        )
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        # Map channels to standard 61-channel configuration
        source_channels = raw.ch_names
        if len(source_channels) != len(STANDARD_61_CHANNELS) or source_channels != STANDARD_61_CHANNELS:
            data_mapped, mapped_channels = map_channels_to_standard_61(
                raw.get_data(),
                source_channels,
                STANDARD_61_CHANNELS,
                self.electrode_set
            )
            
            # Create new Raw object with mapped channels
            info = mne.create_info(
                ch_names=mapped_channels,
                sfreq=raw.info['sfreq'],
                ch_types=['eeg'] * len(mapped_channels)
            )
            raw = mne.io.RawArray(data_mapped, info, verbose=False)
            
            print(f"  [channel mapping] {len(source_channels)} -> {len(mapped_channels)} channels")

        return raw

    def _read_raw(self, vhdr_path: Path) -> mne.io.BaseRaw:
        if not HAS_MNE:
            raise ImportError("MNE is required for MPI-LEMON reader")

        bids_id = vhdr_path.parent.parent.name  # e.g., sub-032528

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_brainvision(
                    str(vhdr_path), preload=True, verbose=False
                )
        except FileNotFoundError as e:
            # Typical for some subjects where VHDR still points to old Initial_ID
            # files (e.g., sub-0103xx.eeg/.vmrk) that are not present. In that case,
            # fall back to the preprocessed EEGLAB .set files.
            print(
                f"  [warn] BrainVision files missing for {bids_id}: {e}. "
                f"Trying preprocessed EEGLAB fallback."
            )
            return self._read_preprocessed_eeglab(bids_id)

        # Drop non-EEG channels (EOG, ECG etc.) heuristically by type
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
        raw.pick(eeg_picks)

        # Ensure units are Volts
        data = raw.get_data()
        data_v, unit = detect_unit_and_convert_to_volts(data)
        if unit != "V":
            raw._data = data_v

        # Bandpass + notch + resample
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            verbose=False,
        )
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)

        # Map channels to standard 61-channel configuration
        source_channels = raw.ch_names
        if len(source_channels) != len(STANDARD_61_CHANNELS) or source_channels != STANDARD_61_CHANNELS:
            data_mapped, mapped_channels = map_channels_to_standard_61(
                raw.get_data(),
                source_channels,
                STANDARD_61_CHANNELS,
                self.electrode_set
            )
            
            # Create new Raw object with mapped channels
            info = mne.create_info(
                ch_names=mapped_channels,
                sfreq=raw.info['sfreq'],
                ch_types=['eeg'] * len(mapped_channels)
            )
            raw = mne.io.RawArray(data_mapped, info, verbose=False)
            
            print(f"  [channel mapping] {len(source_channels)} -> {len(mapped_channels)} channels")

        return raw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_segment(self, seg_uv: np.ndarray) -> bool:
        """Amplitude-based validation in µV."""
        return float(np.abs(seg_uv).max()) <= self.max_amplitude_uv

    # ------------------------------------------------------------------
    # Build per-label HDF5
    # ------------------------------------------------------------------
    def _build_for_subject_and_label(
        self,
        bids_id: str,
        raw: mne.io.BaseRaw,
        labels: Dict[str, Optional[int]],
        label_name: str,
    ) -> Optional[str]:
        """
        Build one HDF5 file for a given subject and label variant.
        """
        label_cfg = self.label_configs[label_name]
        label_idx = labels.get(label_name)
        if label_idx is None:
            # Missing label → skip this subject for this task
            print(f"  [skip] {bids_id}: no label for task '{label_name}'")
            return None

        # Data in V, convert to µV
        data_v = raw.get_data()
        data_uv = data_v * 1e6

        n_ch, n_samp = data_uv.shape

        # Verify channels are standardized to 61 channels
        if list(raw.ch_names) != STANDARD_61_CHANNELS:
            raise ValueError(
                f"Expected {len(STANDARD_61_CHANNELS)} standard channels, "
                f"got {len(raw.ch_names)}. Channel mapping should have been applied earlier."
            )
        
        # Use standard 61-channel configuration
        ch_names = STANDARD_61_CHANNELS
        
        # Windowing
        segments: List[Tuple[np.ndarray, float]] = []
        total_segments = 0
        for start in range(0, n_samp - self.window_samples + 1, self.stride_samples):
            end = start + self.window_samples
            if end > n_samp:
                break
            seg = data_uv[:, start:end]
            
            # Validate segment shape
            if seg.shape[1] != self.window_samples:
                continue
            
            total_segments += 1
            if not self._validate_segment(seg):
                self._stats_per_label[label_name]["rejected_segments"] += 1
                continue
            
            self._stats_per_label[label_name]["valid_segments"] += 1
            segments.append((seg, self.window_sec))
        
        self._stats_per_label[label_name]["total_segments"] += total_segments

        if not segments:
            print(f"  [warn] {bids_id}: no valid segments for task '{label_name}'")
            return None

        # Subject & dataset info
        info = MPI_LEMON_BASE_INFO
        dataset_info = DatasetInfo(
            dataset_name=f"{info.dataset_name}_{label_name}",
            task_type=info.task_type,
            downstream_task_type=info.downstream_task_type,
            num_labels=len(label_cfg.category_list),
            category_list=label_cfg.category_list,
            sampling_rate=self.target_sfreq,
            montage=info.montage,
            channels=STANDARD_61_CHANNELS,
        )

        # Subject attributes
        subject_attrs = SubjectAttrs(
            subject_id=bids_id,
            dataset_name=dataset_info.dataset_name,
            task_type=dataset_info.task_type.value,
            downstream_task_type=dataset_info.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=STANDARD_61_CHANNELS,
            num_labels=dataset_info.num_labels,
            category_list=dataset_info.category_list,
            chn_type="EEG",
            montage=dataset_info.montage,
        )

        # Output path per label variant
        out_dir = self.output_root / dataset_info.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bids_id}.h5"

        # Single trial with many 1s segments
        with HDF5Writer(str(out_path), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=1, session_id=1)
            trial_name = writer.add_trial(trial_attrs)

            label_arr = np.array([label_idx], dtype=np.int64)

            for seg_id, (seg_data, time_len) in enumerate(segments):
                seg_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=seg_id * self.window_sec,
                    end_time=(seg_id + 1) * self.window_sec,
                    time_length=time_len,
                    label=label_arr,
                )
                writer.add_segment(trial_name, seg_attrs, seg_data)

        print(
            f"  [ok] {bids_id}: task='{label_name}', "
            f"segments={len(segments)}, out='{out_path}'"
        )
        return str(out_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_subject(self, bids_id: str) -> Dict[str, Optional[str]]:
        """
        Build all label-variant HDF5s for a single subject.

        Returns:
            dict: {label_name: output_path_or_None}
        """
        vhdr_path = self._find_raw_file(bids_id)
        print(f"\nSubject {bids_id} → {vhdr_path.name}")

        try:
            raw = self._read_raw(vhdr_path)
        except Exception as e:
            print(f"  [error] failed to read raw for {bids_id}: {e}")
            return {name: None for name in self.label_configs}

        labels = self._get_labels_for_subject(bids_id)

        outputs: Dict[str, Optional[str]] = {}
        for label_name in self.label_configs.keys():
            outputs[label_name] = self._build_for_subject_and_label(
                bids_id, raw, labels, label_name
            )
        return outputs

    def _save_dataset_info(self, label_name: str, stats: dict):
        """Save dataset info and processing parameters to JSON for a label variant."""
        label_cfg = self.label_configs[label_name]
        dataset_name = f"{MPI_LEMON_BASE_INFO.dataset_name}_{label_name}"
        
        # Use standard 61-channel configuration
        channels = STANDARD_61_CHANNELS
        
        info = {
            "dataset": {
                "name": dataset_name,
                "description": f"MPI-LEMON Resting-state EEG Dataset - {label_name} classification",
                "task_type": str(MPI_LEMON_BASE_INFO.task_type.value),
                "downstream_task": str(MPI_LEMON_BASE_INFO.downstream_task_type.value),
                "num_labels": len(label_cfg.category_list),
                "category_list": label_cfg.category_list,
                "original_sampling_rate": 500.0,  # MPI-LEMON original sampling rate
                "channels": channels,
                "montage": MPI_LEMON_BASE_INFO.montage,
                "source_url": "https://openneuro.org/datasets/ds000221",
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

        output_dir = self.output_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "dataset_info.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_all(self) -> Dict[str, List[str]]:
        """
        Build HDF5s for all subjects and all label variants.

        Returns:
            dict: {label_name: [list of output paths]}
        """
        # Reset statistics
        self._stats_per_label = {
            name: {
                "total_segments": 0,
                "valid_segments": 0,
                "rejected_segments": 0,
            }
            for name in self.label_configs
        }
        
        all_outputs: Dict[str, List[str]] = {
            name: [] for name in self.label_configs
        }

        subject_ids = self.get_subject_ids()
        print(f"Found {len(subject_ids)} subjects with RSEEG data.")

        for bids_id in subject_ids:
            out_dict = self.build_subject(bids_id)
            for label_name, path in out_dict.items():
                if path:
                    all_outputs[label_name].append(path)

        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        
        # Save dataset info JSON for each label variant
        for label_name, paths in all_outputs.items():
            print(f"\nLabel variant: {label_name}")
            print(f"  Total subjects: {len(subject_ids)}")
            print(f"  Successful subjects: {len(paths)}")
            print(f"  Failed subjects: {len(subject_ids) - len(paths)}")
            
            # Prepare statistics
            label_stats = self._stats_per_label[label_name]
            print(f"  Total segments: {label_stats['total_segments']}")
            print(f"  Valid segments: {label_stats['valid_segments']}")
            print(f"  Rejected segments: {label_stats['rejected_segments']}")
            
            rejection_rate = 0.0
            if label_stats["total_segments"] > 0:
                rejection_rate = label_stats['rejected_segments'] / label_stats['total_segments'] * 100
                print(f"  Rejection rate: {rejection_rate:.1f}%")
            
            print(
                f"  Output directory: {self.output_root / (MPI_LEMON_BASE_INFO.dataset_name + '_' + label_name)}"
            )
            
            stats = {
                "total_subjects": len(subject_ids),
                "successful_subjects": len(paths),
                "failed_subjects": len(subject_ids) - len(paths),
                "total_segments": label_stats["total_segments"],
                "valid_segments": label_stats["valid_segments"],
                "rejected_segments": label_stats["rejected_segments"],
                "rejection_rate": f"{rejection_rate:.1f}%" if label_stats["total_segments"] > 0 else "0%",
            }
            
            # Save dataset info JSON
            self._save_dataset_info(label_name, stats)
        
        print("=" * 50)

        return all_outputs


def build_mpilemon_all(
    raw_root: str,
    beh_root: str,
    output_root: str = "./hdf5",
    target_sfreq: float = 200.0,
    window_sec: float = 1.0,
    stride_sec: float = 1.0,
    filter_low: float = 0.1,
    filter_high: float = 75.0,
    filter_notch: float = 50.0,
    max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
) -> Dict[str, List[str]]:
    """
    Convenience function to build all label variants for all subjects.
    """
    builder = MPILemonBuilder(
        raw_root=raw_root,
        beh_root=beh_root,
        output_root=output_root,
        target_sfreq=target_sfreq,
        window_sec=window_sec,
        stride_sec=stride_sec,
        filter_low=filter_low,
        filter_high=filter_high,
        filter_notch=filter_notch,
        max_amplitude_uv=max_amplitude_uv,
    )
    return builder.build_all()


def main():
    parser = argparse.ArgumentParser(
        description="Build MPI-LEMON resting-state EEG HDF5 datasets "
        "(multiple label variants: gender, age_group)."
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Path to EEG raw root, e.g. "
        "/mnt/dataset2/Datasets/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID",
    )
    parser.add_argument(
        "--beh_root",
        required=True,
        help="Path to behavioural data root, e.g. "
        "/mnt/dataset2/Datasets/MPI-LEMON/Behavioural_Data_MPILMBB_LEMON",
    )
    parser.add_argument(
        "--output_root",
        default="./hdf5",
        help="Output root directory for HDF5 files "
        "(default: ./hdf5, will create subfolders per task)",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=200.0,
        help="Target sampling rate (default: 200.0 Hz)",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=1.0,
        help="Window length in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=1.0,
        help="Stride length in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--filter_low",
        type=float,
        default=0.1,
        help="Low cutoff frequency for bandpass (default: 0.1 Hz)",
    )
    parser.add_argument(
        "--filter_high",
        type=float,
        default=75.0,
        help="High cutoff frequency for bandpass (default: 75.0 Hz)",
    )
    parser.add_argument(
        "--filter_notch",
        type=float,
        default=50.0,
        help="Notch frequency (default: 50.0 Hz, Europe)",
    )
    parser.add_argument(
        "--max_amplitude_uv",
        type=float,
        default=DEFAULT_MAX_AMPLITUDE_UV,
        help="Amplitude threshold in µV for segment rejection "
        f"(default: {DEFAULT_MAX_AMPLITUDE_UV})",
    )

    args = parser.parse_args()

    build_mpilemon_all(
        raw_root=args.raw_root,
        beh_root=args.beh_root,
        output_root=args.output_root,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )


if __name__ == "__main__":
    main()


