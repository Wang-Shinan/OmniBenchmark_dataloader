"""
TUEV (Temple University Epilepsy Corpus) Dataset Builder.

Directory Structure (Flattened version in /mnt/dataset2/Datasets/TUEV):
root/
  00_epilepsy/aaaaaanr/s001_2003/01_tcp_ar/file.edf
  01_no_epilepsy/aaaaaawu/s002_2010/01_tcp_ar/file.edf

Each file is named like: aaaaaaaq_s004_t000.edf
Subject ID: aaaaaaaq
"""

from pathlib import Path
import re
import numpy as np
from collections import defaultdict
import warnings

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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import ElectrodeSet
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

TUEV_INFO = DatasetInfo(
    dataset_name="TUEV",
    task_type=DatasetTaskType.OTHER, # Abnormality/Epilepsy Detection
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["no_epilepsy", "epilepsy"],
    sampling_rate=200.0,
    montage="10_20",
    channels=ElectrodeSet.Standard_10_20
)

# Standard 10-20 System Channels (21 channels) - aligned with utils.py
STANDARD_CHANNELS = ElectrodeSet.Standard_10_20

class TUEVBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 10.0,
        stride_sec: float = 10.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = None, # Disable amplitude filtering
        clip_threshold: float = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "TUEV"
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.clip_threshold = clip_threshold
        
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.electrode_set = ElectrodeSet()

    def get_subject_ids(self):
        """Get list of subject IDs found in raw_data_dir."""
        files = list(self.raw_data_dir.rglob("*.edf"))
        subject_ids = set()
        for path in files:
            # Extract subject ID (e.g., aaaaaaaq from aaaaaaaq_s004_t000.edf)
            match = re.search(r'([a-z]{8})_', path.name)
            if match:
                subject_ids.add(match.group(1))
        return sorted(list(subject_ids))

    def _find_files(self):
        """Find all EDF files and group by subject."""
        files = list(self.raw_data_dir.rglob("*.edf"))
        subject_map = defaultdict(list)
        
        for path in files:
            # Check path components for split and label
            parts = path.parts
            
            # TUEV structure: .../v3.0.0/00_epilepsy/...
            # No explicit train/eval split in folder structure
            # Defaulting to 'train'
            split = "train"
                
            label_str = "unknown"
            if "00_epilepsy" in parts:
                label_str = "epilepsy"
            elif "01_no_epilepsy" in parts:
                label_str = "no_epilepsy"
                
            if label_str == "unknown":
                print(f"Skipping {path.name}: unknown label")
                continue

            # Parse filename for Subject ID
            # aaaaaaaq_s004_t000.edf -> aaaaaaaq
            stem = path.stem
            sub_id_match = re.match(r'([a-zA-Z0-9]+)_s\d+_t\d+', stem)
            if sub_id_match:
                sub_id = sub_id_match.group(1)
            else:
                # Fallback: take first part before _
                sub_id = stem.split('_')[0]
                
            subject_map[sub_id].append({
                "path": path,
                "split": split,
                "label": label_str,
                "stem": stem
            })
            
        return subject_map

    def _process_file(self, file_info):
        """Read and preprocess a single EDF file."""
        path = file_info["path"]
        if not HAS_MNE:
            return None, None
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

            print(f"DEBUG: {path.name} - n_times={raw.n_times}, sfreq={raw.info['sfreq']}, duration={raw.n_times/raw.info['sfreq']:.2f}s")
            
            # Select channels (Standard 10-20 system)
            # Normalize channel names: remove "EEG " prefix, "-REF" suffix, upper case, and standardize (e.g., T3 -> T7)
            original_ch_names = raw.ch_names
            ch_map = {} # Maps standardized name -> original name
            
            for ch in original_ch_names:
                # 1. Clean name
                clean_name = ch.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
                # 2. Standardize (alias mapping)
                std_name = self.electrode_set.standardize_name(clean_name)
                ch_map[std_name] = ch
                
            # Find which standard channels are present
            missing_channels = []
            selected_original_channels = []
            
            for std_ch in STANDARD_CHANNELS:
                # std_ch is already standardized (e.g. T7, T8, P7, P8)
                
                if std_ch in ch_map:
                    selected_original_channels.append(ch_map[std_ch])
                else:
                    missing_channels.append(std_ch)
            
            if len(missing_channels) > 0:
                print(f"  Warning: {path.name} missing channels: {missing_channels}. Skipping.")
                return None, None
                
            # Pick only the standard channels in the correct order
            try:
                raw.pick_channels(selected_original_channels)
                
                # Check current order after pick
                current_std_order = []
                for ch in raw.ch_names:
                    clean = ch.upper().replace("EEG ", "").replace("-REF", "").strip()
                    std = self.electrode_set.standardize_name(clean)
                    current_std_order.append(std)
                    
                if current_std_order != STANDARD_CHANNELS:
                     raw.reorder_channels(selected_original_channels)
            except Exception as e:
                print(f"  Error picking channels for {path.name}: {e}")
                return None, None

            # Rename channels to standard names for consistency in HDF5
            rename_dict = {}
            for orig in selected_original_channels:
                 clean = orig.upper().replace("EEG ", "").replace("-REF", "").strip()
                 std = self.electrode_set.standardize_name(clean)
                 rename_dict[orig] = std
            
            raw.rename_channels(rename_dict)

            # Preprocessing
            if self.filter_notch > 0:
                try:
                    raw.notch_filter(freqs=self.filter_notch, verbose=False)
                except: pass
                
            # Resample
            if raw.info["sfreq"] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)
                
            # Unit conversion to uV
            data = raw.get_data()
            # MNE is in Volts. Check range.
            if np.abs(data).max() < 1.0: # Likely Volts
                data = data * 1e6
            elif np.abs(data).max() < 1000.0: # Likely mV ?? Unlikely for MNE
                data = data * 1e3
            
            if self.clip_threshold is not None:
                data = np.clip(data, -self.clip_threshold, self.clip_threshold)
                
            return data, raw.ch_names
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            return None, None

    def build_subject(self, sub_id):
        """Build a single subject."""
        if not HAS_MNE:
            raise ImportError("MNE required")
            
        subject_map = self._find_files()
        if sub_id not in subject_map:
            raise ValueError(f"Subject {sub_id} not found")
            
        files = subject_map[sub_id]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine subject split (assume all files for a subject are in same split)
        split = files[0]["split"]
        label_str = files[0]["label"] # Assume subject label consistency
        
        # Create HDF5
        out_path = self.output_dir / f"sub_{sub_id}.h5"
        # Overwrite existing files to ensure parameters (like window size) are updated
        if out_path.exists():
            print(f"Overwriting existing {sub_id}...")
            out_path.unlink()
            
        print(f"Processing Subject {sub_id} ({split}, {label_str})...")
        
        subject_attrs = SubjectAttrs(
            subject_id=sub_id,
            dataset_name=f"TUEV_{split}",
            task_type="epilepsy_detection",
            downstream_task_type="classification",
            rsFreq=self.target_sfreq,
            chn_name=STANDARD_CHANNELS, # Use standard channels
            num_labels=2,
            category_list=["no_epilepsy", "epilepsy"],
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20"
        )
        
        with HDF5Writer(str(out_path), subject_attrs) as writer:
            # Sort files by session and trial to ensure deterministic order
            files.sort(key=lambda x: x["stem"])
            
            for unique_trial_id, f_info in enumerate(files):
                data, _ = self._process_file(f_info)
                if data is None:
                    continue
                    
                # Trial attributes
                # Use filename as trial info
                trial_id_match = re.search(r't(\d+)', f_info["stem"])
                # original_trial_num = int(trial_id_match.group(1)) if trial_id_match else 0
                
                sess_id_match = re.search(r's(\d+)', f_info["stem"])
                sess_num = int(sess_id_match.group(1)) if sess_id_match else 0
                
                trial_attrs = TrialAttrs(
                    trial_id=unique_trial_id, # Use unique incremental ID to avoid collisions
                    session_id=sess_num,
                    task_name=f_info["label"]
                )
                trial_name = writer.add_trial(trial_attrs)
                
                # Segment
                n_samples = data.shape[1]
                seg_idx = 0
                label_val = 1 if f_info["label"] == "epilepsy" else 0
                
                for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                    end = start + self.window_samples
                    segment = data[:, start:end]
                    
                    # Validate Amplitude
                    if self.max_amplitude_uv is not None and np.abs(segment).max() > self.max_amplitude_uv:
                        continue
                        
                    seg_attrs = SegmentAttrs(
                        segment_id=seg_idx,
                        label=label_val,
                        start_time=start / self.target_sfreq,
                        end_time=end / self.target_sfreq,
                        time_length=(end - start) / self.target_sfreq,
                        task_label=f_info["label"]
                    )
                    writer.add_segment(trial_name, seg_attrs, segment)
                    seg_idx += 1
        return str(out_path)

    def build(self):
        """Build the dataset."""
        if not HAS_MNE:
            raise ImportError("MNE required")
            
        subject_map = self._find_files()
        print(f"Found {len(subject_map)} subjects.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for sub_id, files in subject_map.items():
            # Determine subject split (assume all files for a subject are in same split)
            split = files[0]["split"]
            label_str = files[0]["label"] # Assume subject label consistency
            
            # Create HDF5
            out_path = self.output_dir / f"sub_{sub_id}.h5"
            # Overwrite existing files to ensure parameters (like window size) are updated
            if out_path.exists():
                print(f"Overwriting existing {sub_id}...")
                out_path.unlink()
                
            print(f"Processing Subject {sub_id} ({split}, {label_str})...")
            
            subject_attrs = SubjectAttrs(
                subject_id=sub_id,
                dataset_name=f"TUEV_{split}",
                task_type="epilepsy_detection",
                downstream_task_type="classification",
                rsFreq=self.target_sfreq,
                chn_name=STANDARD_CHANNELS, # Use standard channels
                num_labels=2,
                category_list=["no_epilepsy", "epilepsy"],
                chn_pos=None,
                chn_ori=None,
                chn_type="EEG",
                montage="10_20"
            )
            
            with HDF5Writer(str(out_path), subject_attrs) as writer:
                # Sort files by session and trial to ensure deterministic order
                files.sort(key=lambda x: x["stem"])
                
                for unique_trial_id, f_info in enumerate(files):
                    data, _ = self._process_file(f_info)
                    if data is None:
                        continue
                        
                    # Trial attributes
                    # Use filename as trial info
                    trial_id_match = re.search(r't(\d+)', f_info["stem"])
                    # original_trial_num = int(trial_id_match.group(1)) if trial_id_match else 0
                    
                    sess_id_match = re.search(r's(\d+)', f_info["stem"])
                    sess_num = int(sess_id_match.group(1)) if sess_id_match else 0
                    
                    trial_attrs = TrialAttrs(
                        trial_id=unique_trial_id, # Use unique incremental ID to avoid collisions
                        session_id=sess_num,
                        task_name=f_info["label"]
                    )
                    trial_name = writer.add_trial(trial_attrs)
                    
                    # Segment
                    n_samples = data.shape[1]
                    seg_idx = 0
                    label_val = 1 if f_info["label"] == "epilepsy" else 0
                    
                    for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                        end = start + self.window_samples
                        segment = data[:, start:end]
                        
                        # Validate Amplitude
                        if self.max_amplitude_uv is not None and np.abs(segment).max() > self.max_amplitude_uv:
                            continue
                            
                        seg_attrs = SegmentAttrs(
                            segment_id=seg_idx,
                            start_time=start / self.target_sfreq,
                            end_time=end / self.target_sfreq,
                            time_length=self.window_sec,
                            label=np.array([label_val])
                        )
                        writer.add_segment(trial_name, seg_attrs, segment)
                        seg_idx += 1

if __name__ == "__main__":
    builder = TUEVBuilder(
        raw_data_dir="/mnt/dataset2/Datasets/TUEV/v3.0.0",
        output_dir="/mnt/dataset2/benchmark_dataloader/hdf5",
        clip_threshold=80.0,
    )
    builder.build()
