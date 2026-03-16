"""
Dataset Builder for MODMA ERP.
Path: datasets/modma_erp.py
"""
import pandas as pd
import numpy as np
import mne
from pathlib import Path
from builder import EEGDatasetBuilder

class MODMAERPBuilder(EEGDatasetBuilder):
    """
    MODMA Dataset (ERP Task).
    Reads EGI .raw continuous files directly.
    """
    
    LABEL_MAP = {
        "HC": 1, "Healthy": 1, "Control": 1,
        "MDD": 2, "Depression": 2, "Patient": 2
    }

    def __init__(self, raw_data_dir, output_dir, dataset_info, preproc_config):
        preproc_config.output_dir = output_dir
        super().__init__(raw_data_dir=raw_data_dir, dataset_info=dataset_info, preproc_config=preproc_config)
        self.output_dir = Path(output_dir)
        self.label_dict = {} 
        self.file_map = {}

    def load_meta_data(self):
        if self.label_dict: return
        
        print("🔍 Searching for subject information Excel file...")
        root_search = self.raw_data_dir.parent
        
        # 搜索 Excel
        potential_files = list(root_search.rglob("*subjects_information*ERP*.xlsx"))
        if not potential_files:
            potential_files = list(root_search.rglob("*subjects_information*.xlsx"))
            
        if not potential_files: 
            print(f"⚠️ Warning: No Excel file found in {root_search}!")
            return
        
        target_xlsx = potential_files[0]
        for f in potential_files:
            if "clean" in f.name: 
                target_xlsx = f
                break
        
        try:
            print(f"✅ Found metadata file: {target_xlsx.name}")
            df = pd.read_excel(target_xlsx, engine='openpyxl')
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            id_col = 'subject id' if 'subject id' in df.columns else df.columns[0]
            type_col = 'type' if 'type' in df.columns else df.columns[1]

            for _, row in df.iterrows():
                try:
                    val = row[id_col]
                    if pd.isna(val): continue
                    sid = str(int(val)) 
                    
                    raw_type = str(row[type_col]).strip()
                    if raw_type in self.LABEL_MAP:
                        self.label_dict[sid] = self.LABEL_MAP[raw_type]
                except Exception: 
                    continue
            print(f"📊 Loaded labels for {len(self.label_dict)} subjects.")
            
        except Exception as e:
            print(f"❌ Failed to load Excel: {e}")

    def get_subject_ids(self) -> list:
        # 扫描文件
        files = sorted(list(self.raw_data_dir.glob("**/*.raw")))
        if not files:
            files = sorted(list(self.raw_data_dir.glob("**/*.mff")))
            
        ids = []
        for f in files:
            try:
                # 提取前8位作为ID (02010002 -> 2010002)
                raw_id_str = f.stem[:8] 
                if raw_id_str.isdigit():
                    norm_id = str(int(raw_id_str))
                    self.file_map[norm_id] = f
                    ids.append(norm_id)
            except: continue
        
        if not ids:
            print(f"⚠️ Warning: No .raw files found in {self.raw_data_dir}")
            
        return ids

    # 必须实现的方法 1
    def get_raw_file_path(self, subject_id) -> Path:
        if not self.file_map: self.get_subject_ids()
        return self.file_map.get(str(subject_id))

    # 必须实现的方法 2 (你刚才报错就是缺这个!)
    def get_trial_info(self, subject_id) -> list[dict]:
        self.load_meta_data()
        label = self.label_dict.get(str(subject_id), 0)
        # 返回默认的一个 trial 信息
        return [{'trial_id': 0, 'session_id': 1, 'start_sec': 0, 'end_sec': 0, 'label': label}]

    def build_subject(self, subject_id: int) -> str:
        from hdf5_io import HDF5Writer
        from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
        
        self.load_meta_data()
        
        file_path = self.get_raw_file_path(subject_id)
        if not file_path: return None
        
        try:
            # 读取原始数据
            raw = mne.io.read_raw_egi(str(file_path), preload=True, verbose=False)
        except Exception as e:
            print(f"❌ Failed to read raw EGI: {e}")
            return None

        # 通道处理
        old_names = raw.ch_names
        if len(old_names) > 128:
            raw.pick(old_names[:128])
            
        new_mapping = {ch: f"E{i+1}" for i, ch in enumerate(raw.ch_names)}
        raw.rename_channels(new_mapping)
        
        # 预处理 (多核加速)
        if self.preproc_config.filter_notch > 0:
            try: 
                raw.notch_filter(self.preproc_config.filter_notch, n_jobs=-1, verbose=False)
            except: pass
            
        raw.filter(self.preproc_config.filter_low, self.preproc_config.filter_high, n_jobs=-1, verbose=False)
        
        if raw.info['sfreq'] != self.preproc_config.target_sfreq:
            raw.resample(self.preproc_config.target_sfreq, n_jobs=-1, verbose=False)
            
        # 数据准备
        processed_data = raw.get_data()
        if np.abs(processed_data).max() < 1e-3:
            processed_data = processed_data * 1e6

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"
        
        label = self.label_dict.get(str(subject_id), 0)
        
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="MODMA_ERP",
            task_type="erp",
            downstream_task_type="classification",
            rsFreq=self.preproc_config.target_sfreq,
            chn_name=raw.ch_names,
            num_labels=2,
            category_list=["Healthy Controls", "Major Depressive Disorder"],
            chn_pos=None, chn_ori=None, chn_type="EEG", montage="HydroCel GSN 128"
        )

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            trial_attrs = TrialAttrs(trial_id=0, session_id=1)
            trial_name = writer.add_trial(trial_attrs)
            
            total_duration = processed_data.shape[1] / self.preproc_config.target_sfreq
            segments = self.segment_trial(processed_data, 0, total_duration)
            
            for seg_id, (seg_data, duration) in enumerate(segments):
                segment_attrs = SegmentAttrs(
                    segment_id=seg_id,
                    start_time=seg_id * self.preproc_config.stride_sec,
                    end_time=seg_id * self.preproc_config.stride_sec + duration,
                    time_length=duration,
                    label=np.array([label])
                )
                writer.add_segment(trial_name, segment_attrs, seg_data)

        return str(output_path)