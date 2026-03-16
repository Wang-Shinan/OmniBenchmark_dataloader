"""
Dataset Builder for SEED_FRA (Full Sessions Version).
Path: datasets/seed_fra.py
Logic:
1. Reads Standard Neuroscan (.cnt) files.
2. Processes ALL 3 SESSIONS for each subject.
3. Uses Blind Slicing (15 trials per session).
4. Merges all sessions into one HDF5 file per subject.
5. FIXED: Removed invalid 'label' argument from TrialAttrs.
"""
import numpy as np
import mne
from pathlib import Path
from builder import EEGDatasetBuilder
from hdf5_io import HDF5Writer
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
from tqdm import tqdm

class SEEDFRABuilder(EEGDatasetBuilder):
    # SEED 标准情感标签
    SEED_LABELS = [1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1] 
    LABEL_MAP = {-1: 0, 0: 1, 1: 2}

    def __init__(self, raw_data_dir, output_dir, dataset_info, preproc_config):
        preproc_config.output_dir = output_dir
        super().__init__(raw_data_dir=raw_data_dir, dataset_info=dataset_info, preproc_config=preproc_config)
        self.output_dir = Path(output_dir)
        self.file_map = {}

    def get_raw_file_path(self, subject_id):
        return self.file_map.get(str(subject_id), [])

    def get_trial_info(self, subject_id):
        return []

    def get_subject_ids(self) -> list:
        search_path = self.raw_data_dir
        if (search_path / "French" / "01-EEG-raw").exists():
            search_path = search_path / "French" / "01-EEG-raw"
        elif (search_path / "01-EEG-raw").exists():
            search_path = search_path / "01-EEG-raw"
            
        print(f"🔍 Searching for .cnt files in: {search_path}")
        files = sorted(list(search_path.rglob("*.cnt")))
        
        ids = []
        for f in files:
            sid = f.stem.split('_')[0]
            if sid not in ids:
                ids.append(sid)
                self.file_map[sid] = []
            self.file_map[sid].append(f)
        
        print(f"✅ Found {len(ids)} subjects (Total {len(files)} files).")
        return ids

    def build_subject(self, subject_id: int) -> str:
        file_paths = self.get_raw_file_path(subject_id)
        if not file_paths: return None
        
        # 排序确保 Session 1, 2, 3 顺序
        file_paths.sort(key=lambda x: x.name)
        
        print(f"\n{'='*40}")
        print(f"🚀 [Subject {subject_id}] Processing {len(file_paths)} Sessions...")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"
        
        try:
            temp_raw = mne.io.read_raw_cnt(str(file_paths[0]), preload=False, verbose=False)
            chn_names = temp_raw.ch_names[:62] if len(temp_raw.ch_names) >= 62 else temp_raw.ch_names
        except Exception as e:
            print(f"❌ Initial read error: {e}")
            return None

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="SEED_FRA",
            task_type="emotion_recognition",
            downstream_task_type="classification",
            rsFreq=self.preproc_config.target_sfreq,
            chn_name=chn_names,
            num_labels=3,
            category_list=["Negative", "Neutral", "Positive"],
            chn_pos=None, chn_ori=None, chn_type="EEG", montage="Standard-10-20-Cap"
        )

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            
            for session_idx, fpath in enumerate(file_paths):
                real_session_id = session_idx + 1
                tqdm.write(f"   📂 Reading Session {real_session_id}: {fpath.name}")
                
                try:
                    raw = mne.io.read_raw_cnt(str(fpath), preload=True, verbose=False)
                except Exception as e:
                    tqdm.write(f"   ❌ Error reading {fpath.name}: {e}")
                    continue

                if len(raw.ch_names) >= 62: raw.pick(raw.ch_names[:62])
                
                # 只有第一个 Session 显示详细进度条
                show_pbar = (session_idx == 0)
                
                # 滤波 & 降采样
                # joblib 安装后，n_jobs=-1 会自动生效加速
                raw.filter(self.preproc_config.filter_low, self.preproc_config.filter_high, n_jobs=-1, verbose=show_pbar)
                if raw.info['sfreq'] != self.preproc_config.target_sfreq:
                    raw.resample(self.preproc_config.target_sfreq, n_jobs=-1, verbose=show_pbar)

                # 盲切逻辑
                current_sfreq = raw.info['sfreq']
                total_samples = raw.n_times
                num_trials = 15
                block_size = total_samples // num_trials
                capture_duration = 180 
                capture_samples = int(capture_duration * current_sfreq)
                
                if capture_samples > block_size: capture_samples = int(block_size * 0.9)

                processed_data = raw.get_data()
                if np.abs(processed_data).max() < 0.01: processed_data *= 1e6

                # 写入 Loop
                for i in tqdm(range(num_trials), desc=f"      Slicing Ses-{real_session_id}", leave=False):
                    block_start = i * block_size
                    block_center = block_start + (block_size // 2)
                    start_idx = block_center - (capture_samples // 2)
                    end_idx = block_center + (capture_samples // 2)
                    
                    if start_idx < 0: start_idx = 0
                    if end_idx > processed_data.shape[1]: end_idx = processed_data.shape[1]

                    trial_data = processed_data[:, start_idx:end_idx]
                    total_dur = trial_data.shape[1] / current_sfreq

                    raw_label = self.SEED_LABELS[i]
                    final_label = self.LABEL_MAP.get(raw_label, 1)

                    global_trial_id = (session_idx * 15) + i
                    
                    trial_attrs = TrialAttrs(trial_id=global_trial_id, session_id=real_session_id)
                    trial_name = writer.add_trial(trial_attrs)
                    
                    segments = self.segment_trial(trial_data, 0, total_dur)
                    for seg_id, (seg_data, duration) in enumerate(segments):
                        segment_attrs = SegmentAttrs(
                            segment_id=seg_id,
                            start_time=seg_id * self.preproc_config.stride_sec,
                            end_time=seg_id * self.preproc_config.stride_sec + duration,
                            time_length=duration,
                            label=np.array([final_label]) 
                        )
                        writer.add_segment(trial_name, segment_attrs, seg_data)
        
        print(f"✅ Saved: {output_path.name} (Total 45 Trials)")
        return str(output_path)