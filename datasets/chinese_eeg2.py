"""
ChineseEEG Dataset Builder (EGI-128, Greedy Sampling, Quota: 40)
Hardware: EGI 128 Geodesic Sensor Net (GSN-HydroCel-128)
Sampling: Greedy non-overlapping search (0.1s stride scan, 1.0s jump on hit)
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from builder import EEGDatasetBuilder
from hdf5_io import HDF5Writer
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
import scipy.signal

class ChineseEEGBuilder(EEGDatasetBuilder):
    def __init__(self, raw_data_dir, output_dir, dataset_info, preproc_config, book_name):
            self.book_name = book_name
            
            # 更新数据集名称
            dataset_info.dataset_name = f"ChineseEEG_{book_name}"
            dataset_info.unit = "uV"
            
            # ✅ 修正点 1：必须是 GSN-HydroCel-128
            dataset_info.montage = "GSN-HydroCel-128" 
            
            preproc_config.output_dir = str(output_dir)
            
            super().__init__(raw_data_dir=raw_data_dir, dataset_info=dataset_info, preproc_config=preproc_config)
            
            self.output_dir = Path(output_dir)
            self.vad_frame_duration = 0.01 
            self.root_path = Path(raw_data_dir)
            self.vad_root = self.root_path / "align_output"
            self.eeg_root = self.root_path / "eeg_data_hdf5_reading"

            # ✅ 修正点 2：通道名必须是 E1 - E128
            self.egi_channels = [f"E{i}" for i in range(1, 129)]

    # --- 辅助函数 ---
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
                if data.shape[0] > 128 and data.shape[1] <= 256: data = data.T
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
            
            # 多数票决 (Majority Vote)
            if n_read > n_rest: return 1
            else: return 0
        except: return -1

    # --- 核心构建 ---
    def build_subject(self, sub_short: str) -> int:
        print(f"\n{'='*40}")
        print(f"🚀 [Sub-{sub_short}] Book: {self.book_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_out = self.output_dir / f"sub_{sub_short}.h5"
        
        subject_attrs = SubjectAttrs(
            subject_id=sub_short,
            dataset_name=self.dataset_info.dataset_name,
            task_type="classification",
            downstream_task_type="reading_detection",
            rsFreq=200,
            chn_name=self.egi_channels, # 使用 E1-E128
            num_labels=2,
            category_list=["No-Reading", "Reading"],
            chn_type="EEG", 
            montage="GSN-HydroCel-128" # 对应 EGI 模板
        )

        book_vad_dir = self.vad_root / self.book_name
        if not book_vad_dir.exists():
            print(f"❌ VAD dir missing: {book_vad_dir}")
            return 0

        run_dirs = sorted([d for d in book_vad_dir.iterdir() if d.is_dir() and "run-" in d.name])
        
        total_segments_subject = 0

        with HDF5Writer(str(final_out), subject_attrs) as writer:
            for run_dir in run_dirs:
                run_id = run_dir.name 
                
                # 1. 定位 VAD
                vad_file = run_dir / sub_short / "vad_results_high_fidelity.txt"
                if not vad_file.exists():
                     found = False
                     for p in run_dir.iterdir():
                         if p.is_dir() and p.name.lower() == sub_short.lower():
                             cand = p / "vad_results_high_fidelity.txt"
                             if cand.exists(): vad_file = cand; found = True; break
                     if not found: continue

                # 2. 定位 EEG
                eeg_file = self.eeg_root / f"sub-{sub_short}" / f"ses-{self.book_name}" / "eeg" / \
                           f"sub-{sub_short}_ses-{self.book_name}_task-reading_{run_id}_eeg.h5"
                if not eeg_file.exists(): continue
                
                print(f"   📂 Processing {run_id}...")

                # 3. 读取 VAD
                try:
                    df_vad = pd.read_csv(vad_file, sep=r'\s+')
                    col_name = self._find_binary_column(df_vad)
                    vad_series = pd.to_numeric(df_vad[col_name], errors='coerce').fillna(0)
                except Exception as e:
                    print(f"      ❌ VAD Error: {e}")
                    continue

                # 4. 读取 EEG & 预处理
                data_raw = self._read_h5_eeg(eeg_file)
                if data_raw is None: continue
                
                fs_orig = 1000.0 
                # 单位转换 V -> uV
                if np.max(np.abs(data_raw)) < 0.1: data_uv = data_raw * 1e6
                else: data_uv = data_raw
                
                try:
                    # 滤波 0.1-75Hz + 50Hz Notch
                    sos = scipy.signal.butter(4, [0.1, 75.0], btype='bandpass', fs=fs_orig, output='sos')
                    data_filt = scipy.signal.sosfiltfilt(sos, data_uv, axis=1)
                    b_n, a_n = scipy.signal.iirnotch(50.0, 30.0, fs=fs_orig)
                    data_filt = scipy.signal.filtfilt(b_n, a_n, data_filt, axis=1)
                except: continue

                # 降采样 1000->200
                ds_factor = int(fs_orig / 200)
                data_resampled = data_filt[:, ::ds_factor]
                fs_final = 200

                # 5. 贪婪采样策略 (Greedy Sampling)
                n_read = 0
                n_rest = 0
                target = 40
                
                t_attrs = TrialAttrs(trial_id=run_id, session_id=0)
                t_name = writer.add_trial(t_attrs)
                
                max_sec = data_resampled.shape[1] / fs_final
                curr_sec = 0.0
                seg_id = 0
                
                # 扫描步长 (Scanning Stride)：0.1s 用于寻找最佳切入点
                scan_stride = 0.1
                
                while curr_sec + 1.0 < max_sec:
                    # 检查配额
                    if n_read >= target and n_rest >= target:
                        print(f"      ✅ Quota Full (40/40). Run finished early.")
                        break
                        
                    label = self._parse_vad_label(vad_series, curr_sec)
                    
                    save_this = False
                    
                    # 只有当名额没满时，才保存
                    if label == 1 and n_read < target:
                        n_read += 1
                        save_this = True
                    elif label == 0 and n_rest < target:
                        n_rest += 1
                        save_this = True
                    
                    if save_this:
                        # ✂️ 切片
                        start_pt = int(curr_sec * fs_final)
                        end_pt = start_pt + 200
                        seg_data = data_resampled[:, start_pt:end_pt]
                        
                        # 填充/截断至128通道
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
                        total_segments_subject += 1
                        
                        # 🟢 关键修改：一旦保存，直接跳过 1.0s，保证无重叠
                        curr_sec += 1.0 
                    else:
                        # 🟡 关键策略：如果不符合要求（或名额已满），只前进 0.1s
                        # 这样可以微调窗口，去捕捉下一个可能的有效片段
                        curr_sec += scan_stride
                
                print(f"      📊 Result: {n_read} Read, {n_rest} No-Read.")
        
        return total_segments_subject