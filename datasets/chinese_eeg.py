"""
ChineseEEG Dataset Builder (Smart Stride for Imbalanced Data)
"""
import sys
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    from ..builder import EEGDatasetBuilder
    from ..hdf5_io import HDF5Writer
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..config import DatasetInfo, PreprocConfig, DatasetTaskType, DownstreamTaskType
except ImportError:
    # Direct script execution - add parent to path first
    _parent_dir = str(Path(__file__).resolve().parent.parent)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from builder import EEGDatasetBuilder
    from hdf5_io import HDF5Writer
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from config import DatasetInfo, PreprocConfig, DatasetTaskType, DownstreamTaskType

import h5py
import numpy as np
import pandas as pd
import scipy.signal

class ChineseEEGBuilder(EEGDatasetBuilder):
    def __init__(self, raw_data_dir, output_dir, dataset_info, preproc_config, book_name):
        dataset_info.dataset_name = f"ChineseEEG_{book_name}"
        dataset_info.unit = "uV"
        dataset_info.montage = "BioSemi-128"
        
        preproc_config.output_dir = output_dir
        super().__init__(raw_data_dir=raw_data_dir, dataset_info=dataset_info, preproc_config=preproc_config)
        
        self.output_dir = Path(output_dir)
        self.book_name = book_name
        self.vad_frame_duration = 0.01 # 10ms
        self.root_path = Path(raw_data_dir)
        self.vad_root = self.root_path / "align_output"
        self.eeg_root = self.root_path / "eeg_data_hdf5_reading"

    def get_raw_file_path(self, subject_id): return []
    def get_trial_info(self, subject_id): return []

    def get_subject_ids(self) -> list:
        subs = []
        if self.eeg_root.exists():
            for p in self.eeg_root.iterdir():
                if p.is_dir() and p.name.startswith("sub-"):
                    subs.append(p.name.replace("sub-", ""))
        return sorted(subs)

    def _read_h5_eeg(self, h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                keys = list(f.keys())
                data_key = None
                for k in ['data', 'eeg', 'raw_signal', 'raw']:
                    if k in keys: data_key = k; break
                if data_key is None:
                    for k in keys:
                        if isinstance(f[k], h5py.Dataset): data_key = k; break
                if data_key is None: return None
                
                data = f[data_key][:]
                # 形状修正 (Channels, Time)
                if data.shape[0] > 128 and data.shape[1] <= 200: data = data.T
                elif data.shape[0] > data.shape[1] and data.shape[1] == 128: data = data.T
                return data
        except Exception as e:
            print(f"      ❌ H5 Error: {e}")
            return None

    def _find_binary_column(self, df):
        if 'is_audio' in df.columns: return 'is_audio'
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().unique()
            if len(vals) <= 2 and all(v in [0, 1] for v in vals): return col
        return df.columns[-1]

    def _parse_vad_label(self, vad_series, start_sec, duration=1.0):
        try:
            start_idx = int(start_sec / self.vad_frame_duration)
            end_idx = int((start_sec + duration) / self.vad_frame_duration)
            if start_idx >= len(vad_series): return -1
            segment = vad_series.iloc[start_idx:end_idx]
            if len(segment) == 0: return -1
            
            n_read = (segment == 1).sum()
            n_rest = (segment == 0).sum()
            
            # 严格多数票
            if n_read > n_rest: return 1
            else: return 0
        except: return -1

    def build_subject(self, sub_short: str) -> str:
        print(f"\n{'='*40}")
        print(f"🚀 [Sub-{sub_short}] Book: {self.book_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_out = self.output_dir / f"sub_{sub_short}_{self.book_name}.h5"
        
        subject_attrs = SubjectAttrs(
            subject_id=sub_short,
            dataset_name=self.dataset_info.dataset_name,
            task_type="classification",
            downstream_task_type="reading_detection",
            rsFreq=200,
            chn_name=[str(i) for i in range(128)], 
            num_labels=2,
            category_list=["No-Reading", "Reading"],
            chn_type="EEG", 
            montage="BioSemi-128"
        )

        book_vad_dir = self.vad_root / self.book_name
        run_dirs = sorted([d for d in book_vad_dir.iterdir() if d.is_dir() and "run-" in d.name])
        
        with HDF5Writer(str(final_out), subject_attrs) as writer:
            total_segments_book = 0
            
            for run_dir in run_dirs:
                run_id = run_dir.name 
                
                # 1. 寻找 VAD
                vad_file = run_dir / sub_short / "vad_results_high_fidelity.txt"
                if not vad_file.exists():
                     found = False
                     for p in run_dir.iterdir():
                         if p.is_dir() and p.name.lower() == sub_short.lower():
                             cand = p / "vad_results_high_fidelity.txt"
                             if cand.exists(): vad_file = cand; found = True; break
                     if not found: continue

                # 2. 寻找 EEG
                eeg_file = self.eeg_root / f"sub-{sub_short}" / f"ses-{self.book_name}" / "eeg" / \
                           f"sub-{sub_short}_ses-{self.book_name}_task-reading_{run_id}_eeg.h5"
                if not eeg_file.exists(): continue
                
                print(f"   📂 Processing {run_id}...")

                # 3. 加载 VAD
                try:
                    df_vad = pd.read_csv(vad_file, sep=r'\s+')
                    col_name = self._find_binary_column(df_vad)
                    vad_series = pd.to_numeric(df_vad[col_name], errors='coerce').fillna(0)
                    
                    ones = (vad_series == 1).sum()
                    zeros = (vad_series == 0).sum()
                    print(f"      🩺 VAD: Ones={ones} ({ones*0.01:.1f}s), Zeros={zeros} ({zeros*0.01:.1f}s)")
                except Exception as e:
                    print(f"      ❌ VAD Error: {e}")
                    continue

                # 4. 加载 EEG & 预处理
                data_raw = self._read_h5_eeg(eeg_file)
                if data_raw is None: continue
                
                fs_orig = 1000.0 
                if np.max(np.abs(data_raw)) < 0.1: data_uv = data_raw * 1e6
                else: data_uv = data_raw
                
                try:
                    sos = scipy.signal.butter(4, [0.1, 75.0], btype='bandpass', fs=fs_orig, output='sos')
                    data_filt = scipy.signal.sosfiltfilt(sos, data_uv, axis=1)
                    b_n, a_n = scipy.signal.iirnotch(50.0, 30.0, fs=fs_orig)
                    data_filt = scipy.signal.filtfilt(b_n, a_n, data_filt, axis=1)
                except: continue

                ds_factor = int(fs_orig / 200)
                data_resampled = data_filt[:, ::ds_factor]
                fs_final = 200

                # 5. 智能配额采样 (Smart Quota Sampling)
                n_read = 0
                n_rest = 0
                target = 40
                
                # Extract run number from run_id (e.g., "run-01" -> 1)
                try:
                    run_num = int(run_id.split('-')[-1])
                except:
                    run_num = total_segments_book  # Fallback to sequential number
                
                t_attrs = TrialAttrs(trial_id=run_num, session_id=0, task_name=run_id)
                t_name = writer.add_trial(t_attrs)
                
                max_sec = data_resampled.shape[1] / fs_final
                curr_sec = 0.0
                seg_id = 0
                
                while curr_sec + 1.0 < max_sec:
                    if n_read >= target and n_rest >= target:
                        print(f"      ✅ Quota Full (40/40).")
                        break
                    
                    label = self._parse_vad_label(vad_series, curr_sec)
                    
                    save_this = False
                    stride = 0.1 # 默认高频搜索 (100ms)
                    
                    if label == 1: # Reading
                        if n_read < target:
                            n_read += 1
                            save_this = True
                            stride = 1.0 # 资源丰富，不重叠
                    elif label == 0: # No-Reading
                        if n_rest < target:
                            n_rest += 1
                            save_this = True
                            stride = 0.1 # 🟢 资源稀缺，允许 90% 重叠采样
                    
                    if save_this:
                        start_pt = int(curr_sec * fs_final)
                        end_pt = start_pt + 200
                        seg_data = data_resampled[:, start_pt:end_pt]
                        
                        if seg_data.shape[0] < 128:
                            pad = np.zeros((128 - seg_data.shape[0], seg_data.shape[1]))
                            seg_data = np.vstack([seg_data, pad])
                        elif seg_data.shape[0] > 128:
                            seg_data = seg_data[:128, :]
                            
                        writer.add_segment(t_name, SegmentAttrs(
                            segment_id=seg_id,
                            start_time=curr_sec,
                            end_time=curr_sec+1.0,
                            time_length=1.0,
                            label=np.array([label])
                        ), seg_data)
                        seg_id += 1
                        total_segments_book += 1
                    
                    # 动态步长前进
                    curr_sec += stride
                
                print(f"      📊 Result: {n_read} Read, {n_rest} No-Read.")
        
        return str(final_out)


if __name__ == "__main__":
    import argparse
    # Ensure we can import config module
    try:
        from config import DatasetInfo, PreprocConfig, DatasetTaskType, DownstreamTaskType
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import DatasetInfo, PreprocConfig, DatasetTaskType, DownstreamTaskType

    parser = argparse.ArgumentParser(
        description="Build ChineseEEG reading vs non-reading HDF5 dataset"
    )
    parser.add_argument(
        "raw_data_dir",
        help="Root directory containing align_output and eeg_data_hdf5_reading",
    )
    parser.add_argument(
        "book_name",
        choices=["garnettdream", "littleprince"],
        help="Book name (garnettdream or littleprince)",
    )
    parser.add_argument(
        "--output_dir",
        default="./hdf5",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (e.g., f1 f2 m1 m2). Default: all subjects.",
    )

    args = parser.parse_args()

    # Create dataset info and preproc config
    dataset_info = DatasetInfo(
        dataset_name=f"ChineseEEG_{args.book_name}",
        task_type=DatasetTaskType.COGNITIVE,
        downstream_task_type=DownstreamTaskType.CLASSIFICATION,
        num_labels=2,
        category_list=["No-Reading", "Reading"],
        sampling_rate=200.0,
        montage="BioSemi-128",
        channels=[str(i) for i in range(128)],
    )

    preproc_config = PreprocConfig(
        output_dir=args.output_dir,
        target_sfreq=200.0,
        filter_low=0.1,
        filter_high=75.0,
        filter_notch=50.0,
        max_amplitude_uv=600.0,
    )

    # Create builder
    builder = ChineseEEGBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        dataset_info=dataset_info,
        preproc_config=preproc_config,
        book_name=args.book_name,
    )

    # Get subject IDs
    all_subjects = builder.get_subject_ids()
    if args.subjects:
        subjects_to_process = [s.replace("sub-", "") for s in args.subjects]
        subjects_to_process = [s for s in subjects_to_process if s in all_subjects]
    else:
        subjects_to_process = all_subjects

    print(f"Processing {len(subjects_to_process)} subjects: {subjects_to_process}")

    # Build all subjects
    for sub_id in subjects_to_process:
        try:
            output_path = builder.build_subject(sub_id)
            print(f"✅ Successfully processed {sub_id}: {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {sub_id}: {e}")