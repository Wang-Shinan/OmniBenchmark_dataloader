"""
Workload Dataset Builder.

Dataset: Cognitive Workload Dataset (MATB task)
- Task: Workload Estimation (Easy, Medium, Difficult)
- Sampling Rate: 250 Hz (Typical for such datasets, read from file)
- Channels: Read from file

Processing:
1. Extract Pxx.zip (Subject specific zip)
2. Iterate through session files (e.g. S01_easy.set, S01_easy.fdt)
3. Load using MNE (EEGLAB format)
4. Standardize channel names (Uppercase)
5. Save raw data segments to HDF5
"""

import os
from pathlib import Path
import numpy as np
import zipfile
import tempfile
import shutil

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

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

WORKLOAD_INFO = DatasetInfo(
    dataset_name="Workload_MATB",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=3,
    category_list=["easy", "medium", "difficult"],
    sampling_rate=250.0, # Placeholder, will be updated from data
    channels=[] # Placeholder
)

LABEL_MAPPING = {
    "easy": 0,
    "med": 1,
    "diff": 2
}

class WorkloadBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / WORKLOAD_INFO.dataset_name
        self.target_sfreq = target_sfreq
        
        # Look for the main zip file or extracted folders
        # The user provided path might be the folder containing 4917218.zip
        # or the extracted contents.
        # We assume raw_data_dir contains 4917218.zip or Pxx.zip files.
        self.main_zip = self.raw_data_dir / "4917218.zip"
        self.electrode_set = ElectrodeSet()
        
    def get_subject_ids(self) -> list[int]:
        # P01 to P15
        return list(range(1, 16))

    def build_subject(self, subject_id: int) -> str:
        if not HAS_MNE:
            raise ImportError("MNE is required")
            
        subject_str = f"P{subject_id:02d}" # P01
        
        # We need to access Pxx.zip
        # If main zip exists, we might need to unzip Pxx.zip from it?
        # That's double unzipping. 
        # Strategy: Create a temp dir, unzip Pxx.zip there.
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Get Pxx.zip
            pxx_zip_path = self.raw_data_dir / f"{subject_str}.zip"
            
            if not pxx_zip_path.exists():
                # Check inside main zip
                if self.main_zip.exists():
                    with zipfile.ZipFile(self.main_zip, 'r') as main_z:
                        try:
                            # The structure inside 4917218.zip might be flat or nested
                            # Let's list names first
                            names = main_z.namelist()
                            target_file = None
                            for n in names:
                                if n.endswith(f"/{subject_str}.zip") or n == f"{subject_str}.zip":
                                    target_file = n
                                    break
                            
                            if target_file:
                                main_z.extract(target_file, temp_path)
                                pxx_zip_path = temp_path / target_file
                            else:
                                raise FileNotFoundError(f"{subject_str}.zip not found in {self.main_zip}")
                        except KeyError:
                             raise FileNotFoundError(f"{subject_str}.zip not found in {self.main_zip}")
                else:
                    raise FileNotFoundError(f"Neither {pxx_zip_path} nor {self.main_zip} found.")
            
            # Step 2: Unzip Pxx.zip
            # Pxx.zip contains .set and .fdt files
            with zipfile.ZipFile(pxx_zip_path, 'r') as p_z:
                p_z.extractall(temp_path)
                
            # Step 3: Find .set files
            # Pattern: *easy.set, *med.set, *diff.set
            set_files = list(temp_path.glob("*.set"))
            if not set_files:
                # Try recursive
                set_files = list(temp_path.rglob("*.set"))
                
            if not set_files:
                print(f"Warning: No .set files found for {subject_str}")
                return ""
            
            # Initialize HDF5 Writer
            # We need channel info first. Read one file.
            try:
                first_raw = mne.io.read_epochs_eeglab(set_files[0], verbose=False)
            except ValueError:
                 first_raw = mne.io.read_raw_eeglab(set_files[0], verbose=False, preload=True)
            
            # Standardize channels
            # Extract channel names and ensure Uppercase
            raw_ch_names = [ch.upper() for ch in first_raw.ch_names]
            
            valid_channels = []
            for target_ch in self.electrode_set.Standard_10_20:
                found = False
                for raw_ch in raw_ch_names:
                    clean_ch = raw_ch.replace("EEG ", "").replace("-REF", "").strip()
                    if clean_ch == target_ch:
                        valid_channels.append(target_ch)
                        found = True
                        break
                if not found:
                    pass
             
            if not valid_channels:
                print(f"Warning: No Standard 10-20 channels found for {subject_str}")
                return ""
            
            subject_attrs = SubjectAttrs(
                subject_id=subject_id,
                dataset_name=WORKLOAD_INFO.dataset_name,
                task_type=WORKLOAD_INFO.task_type.value,
                downstream_task_type=WORKLOAD_INFO.downstream_task_type.value,
                rsFreq=self.target_sfreq,
                chn_name=valid_channels,
                chn_type="EEG"
            )
            
            out_file = self.output_dir / f"sub_{subject_id}.h5"
            writer = HDF5Writer(str(out_file), subject_attrs)
            
            # Process each file with a counter
            trial_counter = 0
            for set_file in set_files:
                # Determine label from filename
                fname = set_file.name.lower()
                label = -1
                for k, v in LABEL_MAPPING.items():
                    if k in fname:
                        label = v
                        break
                
                if label == -1:
                    print(f"  Warning: Could not determine label for {set_file.name}. Skipping.")
                    continue
                    
                print(f"Processing {set_file.name} (Label: {label})")
                
                try:
                    # Use read_epochs_eeglab because these files are often stored as epochs in EEGLAB
                    # But wait, are they raw or epochs? 
                    # Usually workload datasets are continuous. But .set can be both.
                    # If read_raw_eeglab fails, try read_epochs_eeglab
                    try:
                        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
                        is_epochs = False
                    except:
                        # Fallback to epochs
                        epochs = mne.io.read_epochs_eeglab(set_file, verbose=False)
                        is_epochs = True
                        
                    if is_epochs:
                        # Preprocessing: Resample -> Filter -> Notch
                        
                        # 1. Resample (Method on Epochs usually works)
                        if epochs.info['sfreq'] != self.target_sfreq:
                            print(f"  Resampling from {epochs.info['sfreq']} to {self.target_sfreq} Hz")
                            epochs.resample(self.target_sfreq)
                        
                        # Get data as numpy array
                        data = epochs.get_data() # (n_epochs, n_channels, n_times)
                        sfreq = self.target_sfreq
                        
                        # 2. Filter (Functional approach on numpy array to avoid AttributeErrors)
                        # Filter: 0.1-75Hz
                        data = mne.filter.filter_data(data, sfreq, l_freq=0.1, h_freq=75.0, verbose=False)
                        # Notch: 50Hz
                        data = mne.filter.notch_filter(data, sfreq, freqs=50.0, verbose=False)

                        original_ch_names = epochs.ch_names
                    else:
                        # Preprocessing: Resample -> Filter -> Notch
                        if raw.info['sfreq'] != self.target_sfreq:
                            print(f"  Resampling from {raw.info['sfreq']} to {self.target_sfreq} Hz")
                            raw.resample(self.target_sfreq)
                        
                        # Filter: 1.0-75Hz (Increased high-pass to 1.0Hz to reduce EOG/blink artifacts in Delta band)
                        raw.filter(l_freq=1.0, h_freq=75.0, verbose=False)
                        # Notch: 50Hz
                        raw.notch_filter(freqs=50.0, verbose=False)

                        data = raw.get_data() # (n_channels, n_times)
                        sfreq = raw.info['sfreq']
                        original_ch_names = raw.ch_names
                        # Reshape to (1, n_channels, n_times) to unify processing
                        data = data[np.newaxis, :, :]

                    # Resample logic handled above by MNE


                    # Channel mapping
                    # We need to map data channels to valid_channels indices
                    # valid_channels matches the order in subject_attrs
                    # But the file might have different order or extra channels
                    
                    # Create a mapping index
                    file_ch_map = []
                    for i, target_ch in enumerate(valid_channels):
                        # Find target_ch in original_ch_names (case insensitive)
                        found = False
                        for j, orig_ch in enumerate(original_ch_names):
                            clean_orig = orig_ch.upper().replace("EEG ", "").replace("-REF", "").strip()
                            if clean_orig == target_ch:
                                file_ch_map.append(j)
                                found = True
                                break
                        if not found:
                             # Missing channel?
                             # Fill with zeros or raise error?
                             # For now, append -1
                             file_ch_map.append(-1)
                    
                    # Create trial
                    trial_attrs = TrialAttrs(trial_id=trial_counter, session_id=0) # Simple trial ID
                    trial_name = writer.add_trial(trial_attrs)
                    trial_counter += 1
                    
                    # Add segments
                    for i in range(data.shape[0]):
                        epoch_data = data[i] # (n_channels, n_times)
                        
                        # Reorder/Select channels
                        new_data = np.zeros((len(valid_channels), epoch_data.shape[1]), dtype=np.float32)
                        for target_idx, src_idx in enumerate(file_ch_map):
                            if src_idx != -1:
                                new_data[target_idx] = epoch_data[src_idx] * 1e6 # Convert to uV
                                
                        seg_attrs = SegmentAttrs(
                            segment_id=i,
                            start_time=0.0, # Relative to trial
                            end_time=new_data.shape[1] / self.target_sfreq,
                            time_length=new_data.shape[1] / self.target_sfreq,
                            label=np.array([label])
                        )
                        
                        writer.add_segment(trial_name, seg_attrs, new_data)
                        
                except Exception as e:
                    print(f"  Error reading {set_file.name}: {e}")
            
            writer.close()
            return str(out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="/mnt/dataset2/hdf5_datasets")
    args = parser.parse_args()
    
    builder = WorkloadBuilder(args.data_dir, args.output_dir)
    
    print(f"Found {len(builder.get_subject_ids())} subjects.")
    
    for sub_id in builder.get_subject_ids():
        builder.build_subject(sub_id)
