"""
TUH EEG Corpus Dataset Builder.

Directory Structure:
/mnt/dataset2/Datasets/TUH/tuh_eeg/tuh_eeg/v2.0.1/edf/
  group/
    subject_id/
      session/
        montage/
          file.edf

Target:
- 60s segments
- Amplitude clipping (optional)
- Standard 10-20 channels
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import json
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

TUH_INFO = DatasetInfo(
    dataset_name="TUH_EEG",
    task_type=DatasetTaskType.OTHER,  # General Pre-training / Unsupervised
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=0,
    category_list=[],
    sampling_rate=200.0,
    montage="10_20",
    channels=ElectrodeSet.Standard_10_20
)

STANDARD_CHANNELS = ElectrodeSet.Standard_10_20

class TUHBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 60.0,  # 60s segments
        stride_sec: float = 60.0,  # No overlap by default for pre-training
        filter_notch: float = 60.0,
        max_amplitude_uv: float = None,
        clip_threshold: float = None, # Amplitude truncation
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "TUH"
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.clip_threshold = clip_threshold
        
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.electrode_set = ElectrodeSet()

    def _find_files(self):
        """Find all EDF files and group by subject."""
        # Search pattern: v2.0.1/edf/group/subject/session/montage/file.edf
        files = list(self.raw_data_dir.rglob("*.edf"))
        subject_map = defaultdict(list)
        
        print(f"Scanning {self.raw_data_dir}...")
        
        for path in files:
            # Parse Subject ID from path or filename
            # Path example: .../v2.0.1/edf/001/aaaaaadw/s001_2003/02_tcp_le/aaaaaadw_s001_t001.edf
            parts = path.parts
            stem = path.stem
            
            # Try to extract subject ID from filename (aaaaaadw_s001_t001)
            # Pattern: 8 chars subject ID + _s + session + ...
            sub_id_match = re.match(r'([a-zA-Z0-9]{8})_', stem)
            
            if sub_id_match:
                sub_id = sub_id_match.group(1)
            else:
                # Fallback: try to find it in path parts (it's usually the parent of session dir)
                # This is heuristic. Let's rely on filename structure for TUH which is quite standard.
                continue

            subject_map[sub_id].append({
                "path": path,
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
                # Preload=True to allow inplace modification
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

            # print(f"DEBUG: {path.name} - n_times={raw.n_times}, sfreq={raw.info['sfreq']}")
            
            # Standardize Channels
            original_ch_names = raw.ch_names
            ch_map = {} 
            
            for ch in original_ch_names:
                clean_name = ch.upper().replace("EEG ", "").replace("-REF", "").strip()
                std_name = self.electrode_set.standardize_name(clean_name)
                ch_map[std_name] = ch
                
            selected_original_channels = []
            for std_ch in STANDARD_CHANNELS:
                if std_ch in ch_map:
                    selected_original_channels.append(ch_map[std_ch])
            
            # Skip if channels are missing (strict 10-20)
            if len(selected_original_channels) < len(STANDARD_CHANNELS):
                # print(f"  Skipping {path.name}: missing channels")
                return None, None
                
            # Pick and Reorder
            try:
                raw.pick_channels(selected_original_channels, ordered=True)
            except Exception:
                return None, None

            # Rename to standard names
            rename_dict = {orig: self.electrode_set.standardize_name(orig.upper().replace("EEG ", "").replace("-REF", "").strip()) 
                           for orig in selected_original_channels}
            raw.rename_channels(rename_dict)

            # Preprocessing
            # 1. Notch Filter
            if self.filter_notch > 0:
                try:
                    raw.notch_filter(freqs=self.filter_notch, verbose=False)
                except: pass
                
            # 2. Resample
            if raw.info["sfreq"] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)
                
            # 3. Unit Conversion (to uV)
            data = raw.get_data()
            if np.abs(data).max() < 1.0: # Volts
                data = data * 1e6
            elif np.abs(data).max() < 1000.0: # mV?
                data = data * 1e3
            
            # 4. Amplitude Clipping (Truncation)
            if self.clip_threshold is not None:
                data = np.clip(data, -self.clip_threshold, self.clip_threshold)
                
            return data, raw.ch_names
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            return None, None

    def build(self):
        """Build the dataset."""
        if not HAS_MNE:
            raise ImportError("MNE required")
            
        subject_map = self._find_files()
        print(f"Found {len(subject_map)} subjects.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit processing for testing? No, process all found.
        
        for sub_id, files in subject_map.items():
            out_path = self.output_dir / f"sub_{sub_id}.h5"
            if out_path.exists():
                out_path.unlink()
                
            # print(f"Processing Subject {sub_id} ({len(files)} files)...")
            
            subject_attrs = SubjectAttrs(
                subject_id=sub_id,
                dataset_name="TUH_EEG",
                task_type="other",
                downstream_task_type="classification",
                rsFreq=self.target_sfreq,
                chn_name=STANDARD_CHANNELS,
                num_labels=0,
                category_list=[],
                chn_pos=None,
                chn_ori=None,
                chn_type="EEG",
                montage="10_20"
            )
            
            with HDF5Writer(str(out_path), subject_attrs) as writer:
                # Sort files to ensure deterministic order
                files.sort(key=lambda x: x["stem"])
                
                valid_segments_count = 0
                
                for unique_trial_id, f_info in enumerate(files):
                    data, _ = self._process_file(f_info)
                    if data is None:
                        continue
                        
                    # Extract Session Info from filename
                    # aaaaaadw_s001_t001
                    sess_match = re.search(r'_s(\d+)_', f_info["stem"])
                    sess_id = int(sess_match.group(1)) if sess_match else 0
                    
                    trial_attrs = TrialAttrs(
                        trial_id=unique_trial_id,
                        session_id=sess_id,
                        task_name=""
                    )
                    trial_name = writer.add_trial(trial_attrs)
                    
                    n_samples = data.shape[1]
                    
                    # Sliding Window
                    for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                        end = start + self.window_samples
                        segment = data[:, start:end]
                        
                        # Validate Amplitude (Rejection)
                        if self.max_amplitude_uv is not None:
                            if np.abs(segment).max() > self.max_amplitude_uv:
                                continue
                                
                        seg_attrs = SegmentAttrs(
                            segment_id=valid_segments_count,
                            start_time=start / self.target_sfreq,
                            end_time=end / self.target_sfreq,
                            time_length=self.window_sec,
                            label=np.array([-1]) # No label
                        )
                        writer.add_segment(trial_name, seg_attrs, segment)
                        valid_segments_count += 1
            
            print(f"Saved {sub_id}: {valid_segments_count} segments")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TUH EEG Corpus HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw TUH files (will search recursively for .edf files)")
    parser.add_argument("--output_dir", default="/mnt/dataset2/hdf5_datasets", help="Output directory")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate (Hz)")
    parser.add_argument("--window_sec", type=float, default=60.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=60.0, help="Stride length in seconds")
    parser.add_argument("--filter_notch", type=float, default=60.0, help="Notch filter frequency (Hz, 0 to disable)")
    parser.add_argument("--max_amplitude_uv", type=float, default=None, help="Maximum amplitude threshold (µV, None to disable)")
    parser.add_argument("--clip_threshold", type=float, default=None, help="Amplitude clipping threshold (µV, None to disable)")
    args = parser.parse_args()

    builder = TUHBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        clip_threshold=args.clip_threshold,
    )
    builder.build()
