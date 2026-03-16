"""
Sleep-EDF Dataset Builder.

Sleep-EDF Database Expanded (sleep-cassette).
- Task: Sleep Staging (5 classes: W, N1, N2, N3, R)
- Sampling Rate: 100 Hz
- Channels: 2 EEG (Fpz-Cz, Pz-Oz) + EOG + EMG + ...

Processing:
1. Load PSG and Hypnogram EDF files using MNE
2. Crop data to match Hypnogram
3. Extract standard EEG channels
4. Segment into 30s non-overlapping windows
5. Map stages to standard 5-class system
"""

import os
from pathlib import Path
import numpy as np
import glob

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


SLEEPEDF_INFO = DatasetInfo(
    dataset_name="SleepEDF",
    task_type=DatasetTaskType.SLEEP,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=5,
    category_list=["W", "N1", "N2", "N3", "R"],
    sampling_rate=100.0,
    montage="standard_1020", # Actually not strictly 10-20, but Fpz-Cz/Pz-Oz are standard locs
    channels=['FPZ', 'PZ']
)

# Mapping from Sleep-EDF annotations to 5 classes
# Sleep stage W -> 0
# Sleep stage 1 -> 1
# Sleep stage 2 -> 2
# Sleep stage 3 -> 3
# Sleep stage 4 -> 3 (N3)
# Sleep stage R -> 4
# Sleep stage ? -> Ignore
STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

class SleepEDFBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "/mnt/dataset2/hdf5_datasets",
        target_sfreq: float = 200.0,
        window_sec: float = 30.0, # Sleep staging typically uses 30s epochs
        stride_sec: float = 30.0,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "SleepEDF"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        
        # Verify electrode names - SleepEDF uses Bipolar channels which are not in standard set
        # But we can check if we want to standardize them. 
        # For now, keep as is but acknowledge they are bipolar.
        self.electrode_set = ElectrodeSet()
        
    def get_subject_ids(self) -> list[int]:
        """
        Scan directory for subjects.
        Returns a list of subject IDs (integers).
        """
        # Pattern: SC4{subject_id:02d}*E0-PSG.edf
        # Get all SC4... files
        files = list(self.raw_data_dir.glob("SC4*E0-PSG.edf"))
        subjects = set()
        for f in files:
            # Extract subject ID. 
            # SC4001E0-PSG.edf -> 00
            # SC4011E0-PSG.edf -> 01
            # SC4ssNE0 -> ss is subject ID
            name = f.name
            if name.startswith("SC4"):
                try:
                    sid = int(name[3:5])
                    subjects.add(sid)
                except ValueError:
                    pass
        return sorted(list(subjects))

    def build_subject(self, subject_id: int) -> str:
        """
        Build HDF5 for a subject. 
        Note: SleepEDF subjects often have 2 nights (files). We can treat them as sessions.
        Subject ID in SleepEDF is like '4001'.
        """
        if not HAS_MNE:
            raise ImportError("MNE is required")
            
        # Find files for this subject
        # Pattern: SC4{subject_id:02d}[12]*.edf
        # e.g. SC400110 -> Subject 00, Night 1. 
        
        # Search for PSG files
        search_pattern = f"SC4{subject_id:02d}*E0-PSG.edf"
        psg_files = sorted(list(self.raw_data_dir.glob(search_pattern)))
        
        if not psg_files:
            raise FileNotFoundError(f"No PSG files found for subject {subject_id} with pattern {search_pattern}")
            
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SLEEPEDF_INFO.dataset_name,
            task_type=SLEEPEDF_INFO.task_type.value,
            downstream_task_type=SLEEPEDF_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=SLEEPEDF_INFO.channels,
            chn_type="EEG"
        )
        
        out_file = self.output_dir / f"sub_{subject_id}.h5"
        writer = HDF5Writer(str(out_file), subject_attrs)
        
        for i, psg_path in enumerate(psg_files):
            # Find corresponding Hypnogram
            # SC4001E0-PSG.edf -> SC4001EC-Hypnogram.edf
            # Pattern is identical except 'PSG' -> 'Hypnogram' and 'E0' -> 'EC' (sometimes)
            # Actually standard names are SC4ssNE0-PSG.edf and SC4ssNEC-Hypnogram.edf
            # But suffix can vary (EC, EH, EJ...), so use glob matching
            prefix = psg_path.name[:6] # e.g. SC4001
            hypno_candidates = list(psg_path.parent.glob(f"{prefix}*Hypnogram.edf"))
            
            if not hypno_candidates:
                print(f"  Warning: Hypnogram not found for {psg_path.name} (prefix {prefix}). Skipping.")
                continue
                
            hypno_path = hypno_candidates[0]
            
            print(f"Processing Subject {subject_id} Night {i+1}: {psg_path.name}")
            
            try:
                # Load Raw
                raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
                annot = mne.read_annotations(hypno_path)
                raw.set_annotations(annot, emit_warning=False)
                
                # Pick channels
                # Standard channels: 'EEG Fpz-Cz', 'EEG Pz-Oz'
                # Map them to 'FPZ', 'PZ'
                channel_map = {
                    'EEG Fpz-Cz': 'FPZ',
                    'EEG Pz-Oz': 'PZ'
                }
                
                available_channels = raw.ch_names
                picks = []
                
                # Check which original channels are present
                for orig_ch in channel_map.keys():
                    if orig_ch in available_channels:
                        picks.append(orig_ch)
                    else:
                        print(f"  Warning: Channel {orig_ch} not found in {psg_path.name}")
                
                if not picks:
                    print(f"  Error: No EEG channels found. Skipping.")
                    continue
                    
                # Pick original channels first
                raw.pick_channels(picks)
                
                # Rename to standard names
                mne.rename_channels(raw.info, channel_map)
                
                # Resample if needed
                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq)
                
                # Filter: 0.1-75Hz
                raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
                # Notch: 50Hz
                raw.notch_filter(freqs=50.0, verbose=False)
                    
                # Extract events from annotations
                events, event_id = mne.events_from_annotations(
                    raw, event_id=STAGE_MAPPING, chunk_duration=30., verbose=False
                )
                
                # Create Epochs
                # tmin=0, tmax=30 - 1/sfreq
                epochs = mne.Epochs(
                    raw, events, event_id, tmin=0, tmax=30. - 1./self.target_sfreq,
                    baseline=None, preload=True, verbose=False
                )
                
                # Save to HDF5
                data = epochs.get_data() * 1e6 # (n_epochs, n_channels, n_times) Convert to uV
                labels = epochs.events[:, 2] # (n_epochs,)
                
                # Create trial (Session)
                trial_attrs = TrialAttrs(trial_id=i, session_id=i)
                trial_name = writer.add_trial(trial_attrs)
                
                for j in range(len(labels)):
                    seg_data = data[j]
                    seg_label = labels[j]
                    
                    seg_attrs = SegmentAttrs(
                        segment_id=j,
                        start_time=j * 30.0,
                        end_time=(j+1) * 30.0,
                        time_length=30.0,
                        label=np.array([seg_label])
                    )
                    
                    writer.add_segment(trial_name, seg_attrs, seg_data)
                    
            except Exception as e:
                print(f"Error processing {psg_path.name}: {e}")
        
        writer.close()
        return str(out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="/mnt/dataset2/hdf5_datasets")
    args = parser.parse_args()
    
    builder = SleepEDFBuilder(args.data_dir, args.output_dir)
    
    print(f"Found {len(builder.get_subject_ids())} subjects.")
    
    for sub_id in builder.get_subject_ids():
        builder.build_subject(sub_id)
