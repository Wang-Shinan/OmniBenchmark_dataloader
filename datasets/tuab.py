"""
TUAB (Temple University Abnormal EEG Corpus) Dataset Builder.

Directory Structure (Flattened version in /mnt/dataset2/Datasets/TUAB):
root/
  train/
    abnormal/01_tcp_ar/file.edf
    normal/01_tcp_ar/file.edf
  eval/
    abnormal/01_tcp_ar/file.edf
    normal/01_tcp_ar/file.edf

Each file is named like: aaaaaaaq_s004_t000.edf
Subject ID: aaaaaaaq
"""

from pathlib import Path
import re
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
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

TUAB_INFO = DatasetInfo(
    dataset_name="TUAB",
    task_type=DatasetTaskType.OTHER, # Abnormality Detection
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["normal", "abnormal"],
    sampling_rate=200.0,
    montage="10_20",
    channels=ElectrodeSet.Standard_10_20
)

# Standard 10-20 System Channels (21 channels) - aligned with utils.py
STANDARD_CHANNELS = ElectrodeSet.Standard_10_20

class TUABBuilder:
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
        self.output_dir = Path(output_dir) / "TUAB"
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.clip_threshold = clip_threshold
        
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.electrode_set = ElectrodeSet()
        
        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        
        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = None

    def _find_files(self):
        """Find all EDF files and group by subject."""
        files = list(self.raw_data_dir.rglob("*.edf"))
        subject_map = defaultdict(list)
        
        for path in files:
            # Check path components for split and label
            # Path expected: .../split/label/01_tcp_ar/filename.edf
            # We look for 'train'/'eval' and 'normal'/'abnormal' in parts
            parts = path.parts
            
            split = "unknown"
            if "train" in parts:
                split = "train"
            elif "eval" in parts:
                split = "eval"
                
            label_str = "unknown"
            if "normal" in parts:
                label_str = "normal"
            elif "abnormal" in parts:
                label_str = "abnormal"
                
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
            return None
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

            # Select channels (Standard 10-20 system)
            # Normalize channel names: remove "EEG " prefix, "-REF" suffix, upper case, and standardize (e.g., T3 -> T7)
            original_ch_names = raw.ch_names
            ch_map = {} # Maps standardized name -> original name
            
            for ch in original_ch_names:
                # 1. Clean name
                clean_name = ch.upper().replace("EEG ", "").replace("-REF", "").strip()
                # 2. Standardize (alias mapping)
                std_name = self.electrode_set.standardize_name(clean_name)
                ch_map[std_name] = ch
                
            # Find which standard channels are present
            missing_channels = []
            selected_original_channels = []
            
            for std_ch in STANDARD_CHANNELS:
                # std_ch is already standardized (e.g. T7, T8, P7, P8)
                # But TUAB files might have T3, T4, T5, T6
                # ch_map keys are standardized.
                
                if std_ch in ch_map:
                    selected_original_channels.append(ch_map[std_ch])
                else:
                    missing_channels.append(std_ch)
            
            if len(missing_channels) > 0:
                print(f"  Warning: {path.name} missing channels: {missing_channels}. Skipping.")
                return None, None
                
            # Pick only the standard channels in the correct order
            try:
                raw.pick(selected_original_channels)
                
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
                
            # Common average reference or other filters could go here
            # For now, just simple filtering
            
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

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available, otherwise use STANDARD_CHANNELS
        channels = self._dataset_channels if self._dataset_channels else STANDARD_CHANNELS
        
        info = {
            "dataset": {
                "name": TUAB_INFO.dataset_name,
                "description": "Temple University Hospital Abnormal EEG Corpus",
                "task_type": str(TUAB_INFO.task_type.value),
                "downstream_task": str(TUAB_INFO.downstream_task_type.value),
                "num_labels": TUAB_INFO.num_labels,
                "category_list": TUAB_INFO.category_list,
                "original_sampling_rate": None,  # TUAB files have variable sampling rates
                "channels": channels,
                "montage": TUAB_INFO.montage,
                "source_url": "https://www.isip.piconepress.com/projects/tuh_eeg/",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "clip_threshold": self.clip_threshold,
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

    def build(self):
        """Build the dataset."""
        if not HAS_MNE:
            raise ImportError("MNE required")
        
        # Reset statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        self._dataset_channels = None
            
        subject_map = self._find_files()
        print(f"Found {len(subject_map)} subjects.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_subjects = []
        failed_subjects = []
        
        for sub_id, files in subject_map.items():
            # Determine subject split (assume all files for a subject are in same split)
            split = files[0]["split"]
            label_str = files[0]["label"] # Assume subject label consistency
            
            # Subject ID needs to be int for HDF5 schema usually, but schema allows Union[int, str]
            # But let's check schema.py. It says Union[int, str].
            # However, hdf5_io might expect int? No, usually fine.
            # But `sub_*.h5` loading logic in loader.py might try to parse int.
            # loader.py: _get_subject_id tries to parse int, if fails returns str.
            
            # Create HDF5
            out_path = self.output_dir / f"sub_{sub_id}.h5"
            # Overwrite existing files to ensure parameters (like window size) are updated
            if out_path.exists():
                print(f"Overwriting existing {sub_id}...")
                out_path.unlink()
                
            print(f"Processing Subject {sub_id} ({split}, {label_str})...")
            
            # Process first file to get channel names before creating HDF5Writer
            ch_names = None
            subject_attrs = None
            first_valid_file_data = None
            first_valid_file_info = None
            
            try:
                # Sort files by session and trial to ensure deterministic order
                files.sort(key=lambda x: x["stem"])
                
                # Find first valid file to get channel names
                for f_info in files:
                    data, processed_ch_names = self._process_file(f_info)
                    if data is not None and processed_ch_names is not None:
                        ch_names = processed_ch_names
                        first_valid_file_data = data
                        first_valid_file_info = f_info
                        # Store channel names for dataset info (first subject sets it)
                        if self._dataset_channels is None:
                            self._dataset_channels = ch_names
                        break
                
                # If no valid file found, skip this subject
                if ch_names is None:
                    print(f"  Subject {sub_id}: No valid files found, skipping")
                    failed_subjects.append(sub_id)
                    continue
                
                # Create subject attributes with actual channel names
                subject_attrs = SubjectAttrs(
                    subject_id=sub_id,
                    dataset_name=f"TUAB_{split}",
                    task_type="abnormality_detection",
                    downstream_task_type="classification",
                    rsFreq=self.target_sfreq,
                    chn_name=ch_names,
                    num_labels=2,
                    category_list=["normal", "abnormal"],
                    chn_pos=None,
                    chn_ori=None,
                    chn_type="EEG",
                    montage="10_20"
                )
                
                # Now create HDF5Writer with proper subject_attrs
                with HDF5Writer(str(out_path), subject_attrs) as writer:
                    # Process all files (including the first one we already processed)
                    for unique_trial_id, f_info in enumerate(files):
                        # Use cached data for first valid file, otherwise process
                        if f_info["path"] == first_valid_file_info["path"]:
                            data = first_valid_file_data
                        else:
                            data, _ = self._process_file(f_info)
                            if data is None:
                                continue
                        
                        # Trial attributes
                        # Use filename as trial info
                        trial_id_match = re.search(r't(\d+)', f_info["stem"])
                        original_trial_num = int(trial_id_match.group(1)) if trial_id_match else 0
                        
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
                        label_val = 1 if f_info["label"] == "abnormal" else 0
                        
                        for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                            end = start + self.window_samples
                            segment = data[:, start:end]
                            
                            self.total_segments += 1
                            
                            # Validate Amplitude
                            if self.max_amplitude_uv is not None and np.abs(segment).max() > self.max_amplitude_uv:
                                self.rejected_segments += 1
                                continue
                            
                            self.valid_segments += 1
                            
                            seg_attrs = SegmentAttrs(
                                segment_id=seg_idx,
                                start_time=start / self.target_sfreq,
                                end_time=end / self.target_sfreq,
                                time_length=self.window_sec,
                                label=np.array([label_val])
                            )
                            writer.add_segment(trial_name, seg_attrs, segment)
                            seg_idx += 1
                
                successful_subjects.append(sub_id)
                print(f"  Subject {sub_id}: {seg_idx} segments")
                    
            except Exception as e:
                print(f"Error processing subject {sub_id}: {e}")
                failed_subjects.append(sub_id)
                import traceback
                traceback.print_exc()
        
        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_map)}")
        print(f"Successful: {len(successful_subjects)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects[:20]}{'...' if len(failed_subjects) > 20 else ''}")
        print(f"\nTotal segments: {self.total_segments}")
        print(f"Valid segments: {self.valid_segments}")
        print(f"Rejected segments: {self.rejected_segments}")
        if self.total_segments > 0:
            print(f"Rejection rate: {self.rejected_segments / self.total_segments * 100:.1f}%")
        print("=" * 50)
        
        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_map),
            "successful_subjects": len(successful_subjects),
            "failed_subjects": failed_subjects,
            "total_segments": self.total_segments,
            "valid_segments": self.valid_segments,
            "rejected_segments": self.rejected_segments,
            "rejection_rate": f"{self.rejected_segments / self.total_segments * 100:.1f}%" if self.total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

if __name__ == "__main__":
    builder = TUABBuilder(
        raw_data_dir="/mnt/dataset2/Datasets/TUAB/tuh_eeg_abnormal/v3.0.1/edf",
        output_dir="/mnt/dataset2/benchmark_dataloader/hdf5",
        clip_threshold=80.0,
    )
    builder.build()
