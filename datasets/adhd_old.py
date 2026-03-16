"""
ADHD Dataset Builder.

Dataset: ADHD Dataset (likely TUMS / Nasrabadi et al.)
- Task: ADHD vs Control Classification
- Sampling Rate: 128 Hz
- Channels: 19 (10-20 standard)

Data Structure:
- ADHD_part1/
  - v1p.mat, v2p.mat...
- Control_part1/
  - v1.mat, v2.mat...

Processing:
1. Load .mat files
2. Transpose to (channels, time)
3. Standardize channel names to 10-20 system (Uppercase)
4. Segment into 2s windows (no overlap by default, or with stride)
"""

import os
from pathlib import Path
import numpy as np
import scipy.io
import glob
import mne

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from ..utils import ElectrodeSet
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType
    from utils import ElectrodeSet

# Standard 10-20 channels in the order appearing in .ced file / data
# Ensure Uppercase to match ElectrodeSet
ADHD_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
    'T7', 'C3', 'CZ', 'C4', 'T8', 
    'P7', 'P3', 'PZ', 'P4', 'P8', 
    'O1', 'O2'
]

ADHD_INFO = DatasetInfo(
    dataset_name="ADHD_TUMS", ## TODO need num_classes?
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["Control", "ADHD"],
    sampling_rate=128.0,
    montage="10_20",
    channels=ADHD_CHANNELS
)

class ADHDBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
        window_sec: float = 2.0,
        stride_sec: float = 2.0,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "ADHD"
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        
        # Verify electrode names
        self.electrode_set = ElectrodeSet()
        self._verify_channels()
        
        self.files_map = self._scan_files()
        
    def _verify_channels(self):
        """Ensure all channels are valid according to standard schema."""
        for ch in ADHD_CHANNELS:
            if not self.electrode_set.is_valid_electrode(ch):
                print(f"Warning: Channel {ch} is not in standard ElectrodeSet!")

    def _scan_files(self):
        """Scan directories and map subject ID to file info."""
        files = []
        
        # ADHD (Label 1)
        for part in ["ADHD_part1", "ADHD_part2"]:
            p = self.raw_data_dir / part
            if p.exists():
                for f in p.glob("*.mat"):
                    files.append({
                        "path": f,
                        "label": 1,
                        "group": "ADHD"
                    })
            else:
                print(f"Warning: Directory {p} does not exist.")
                    
        # Control (Label 0)
        for part in ["Control_part1", "Control_part2"]:
            p = self.raw_data_dir / part
            if p.exists():
                for f in p.glob("*.mat"):
                    files.append({
                        "path": f,
                        "label": 0,
                        "group": "Control"
                    })
            else:
                print(f"Warning: Directory {p} does not exist.")
        
        # Sort by filename to ensure consistency
        files.sort(key=lambda x: x["path"].name)
        
        if not files:
            raise FileNotFoundError(f"No .mat files found in {self.raw_data_dir}")
            
        # Create map: subject_id (0-based index) -> file_info
        return {i: f for i, f in enumerate(files)}

    def get_subject_ids(self) -> list[int]:
        return list(self.files_map.keys())

    def build_subject(self, subject_id: int) -> str:
        if subject_id not in self.files_map:
            raise ValueError(f"Subject ID {subject_id} not found.")
            
        info = self.files_map[subject_id]
        file_path = info["path"]
        label = info["label"]
        
        print(f"Processing Subject {subject_id}: {file_path.name} (Label: {label})")
        
        try:
            mat = scipy.io.loadmat(file_path)
            # Find key. Usually filename without extension.
            # e.g. v10p.mat -> v10p
            key = file_path.stem
            if key not in mat:
                # Fallback: look for any key that is not dunder
                keys = [k for k in mat.keys() if not k.startswith("__")]
                if len(keys) == 1:
                    key = keys[0]
                else:
                    # Specific fix for some files that might have different keys
                    # Just take the variable with the largest size
                    largest_key = max(keys, key=lambda k: mat[k].size)
                    key = largest_key
                    print(f"  Note: Using key '{key}' for {file_path.name}")
            
            data = mat[key] # (time, channels) or (channels, time)
            
            # Check shape
            # We expect 19 channels.
            if data.shape[1] == 19:
                eeg_data = data.T # Convert to (channels, time)
            elif data.shape[0] == 19:
                eeg_data = data
            else:
                print(f"  Error: Unexpected shape {data.shape} for {file_path.name}. Expected 19 channels. Skipping.")
                return ""
                
            # Create HDF5
            subject_attrs = SubjectAttrs(
                subject_id=subject_id,
                dataset_name=ADHD_INFO.dataset_name,
                task_type=ADHD_INFO.task_type.value,
                downstream_task_type=ADHD_INFO.downstream_task_type.value,
                rsFreq=self.target_sfreq,
                chn_name=ADHD_INFO.channels,
                chn_type="EEG"
            )
            
            out_file = self.output_dir / f"sub_{subject_id}.h5"
            writer = HDF5Writer(str(out_file), subject_attrs)
            
            # One trial per file
            trial_attrs = TrialAttrs(trial_id=0, session_id=0)
            trial_name = writer.add_trial(trial_attrs)
            
            # Segmentation
            n_samples = eeg_data.shape[1]
            n_segments = (n_samples - self.window_samples) // self.stride_samples + 1
            
            if n_segments <= 0:
                 print(f"  Warning: Data too short for windowing ({n_samples} samples).")
            else:
                for i in range(n_segments):
                    start = i * self.stride_samples
                    end = start + self.window_samples
                    seg_data = eeg_data[:, start:end]
                    
                    seg_attrs = SegmentAttrs(
                        segment_id=i,
                        start_time=start / self.target_sfreq,
                        end_time=end / self.target_sfreq,
                        time_length=self.window_sec,
                        label=np.array([label])
                    )
                    
                    writer.add_segment(trial_name, seg_attrs, seg_data)
            
            writer.close()
            return str(out_file)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="/mnt/dataset2/hdf5_datasets")
    args = parser.parse_args()
    
    builder = ADHDBuilder(args.data_dir, args.output_dir)
    
    print(f"Found {len(builder.get_subject_ids())} subjects.")
    
    # Process all subjects
    for sub_id in builder.get_subject_ids():
        builder.build_subject(sub_id)
