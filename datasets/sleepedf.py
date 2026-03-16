import numpy as np
import mne
from pathlib import Path
from builder import EEGDatasetBuilder
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
from hdf5_io import HDF5Writer

class SleepEDFBuilder(EEGDatasetBuilder):
    """
    Builder for Sleep-EDF Database Expanded (2018).
    Standard: Sleep Staging (AASM), 30s Epochs, 100Hz.
    Ref: https://www.physionet.org/content/sleep-edfx/1.0.0/
    """
    def __init__(self, dataset_info, preproc_config, raw_data_dir):
        super().__init__(dataset_info, preproc_config, raw_data_dir)
        
        # ⚠️ 睡眠分期标准参数
        self.target_fs = 100.0      # Sleep-EDF 原生就是 100Hz
        self.epoch_len_sec = 30.0   # 必须是 30秒
        
        # AASM 标准映射表
        self.stage_mapping = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,  # N3
            'Sleep stage 4': 3,  # N3 (合并)
            'Sleep stage R': 4
        }
        
        # 目标通道
        self.target_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    def get_subject_ids(self) -> list:
        psg_files = sorted(list(self.raw_data_dir.glob("*PSG.edf")))
        ids = [p.name.split('-')[0] for p in psg_files]
        return ids

    def get_raw_file_path(self, subject_id) -> Path:
        return self.raw_data_dir

    def get_label_file_path(self, subject_id) -> Path:
        return self.raw_data_dir

    def get_trial_info(self, subject_id) -> list[dict]:
        return []

    def build_subject(self, subject_id) -> str:
        # 文件匹配逻辑: SC4001E0-PSG.edf matches SC4001EC-Hypnogram.edf
        prefix = subject_id[:7]
        
        psg_files = list(self.raw_data_dir.glob(f"{subject_id}-PSG.edf"))
        hyp_files = list(self.raw_data_dir.glob(f"{prefix}*-Hypnogram.edf"))
        
        if not psg_files or not hyp_files:
            print(f"❌ 文件不匹配或缺失: {subject_id}")
            return None
            
        psg_path = psg_files[0]
        hyp_path = hyp_files[0]

        try:
            # ================= [Stage 1] 加载与筛选 =================
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            # 提取通道
            available_chs = [ch for ch in self.target_channels if ch in raw.ch_names]
            if not available_chs:
                print(f"❌ {subject_id} 中没找到 EEG Fpz-Cz 或 Pz-Oz")
                return None
            
            # 使用 pick 而非 pick_channels (消除 Legacy Warning)
            raw.pick(available_chs)
            
            # ================= [Stage 2] 预处理 (Sleep Standard) =================
            # 1. 滤波: 0.3-35Hz (睡眠标准)
            # 注意：低通 35Hz 已经把 50Hz 干扰去除了，所以不需要再做 50Hz 陷波
            raw.filter(l_freq=0.3, h_freq=35.0, fir_design='firwin', verbose=False)
            
            # 2. 智能陷波: 只有当陷波频率小于 Nyquist 时才执行
            nyquist = raw.info['sfreq'] / 2.0
            notch_freq = 50.0
            
            if notch_freq < nyquist:
                raw.notch_filter(freqs=notch_freq, verbose=False)
            else:
                # 只有第一个受试者打印提示，避免刷屏
                if subject_id.endswith("001E0") or subject_id.endswith("002E0"):
                    print(f"   ℹ️ 跳过 50Hz 陷波 (采样率{raw.info['sfreq']}Hz, Nyquist {nyquist}Hz, 已被35Hz低通覆盖)")

            # 3. 重采样
            if raw.info['sfreq'] != self.target_fs:
                raw.resample(self.target_fs)
            
            # ================= [Stage 3] 读取标签与切片 =================
            annot = mne.read_annotations(str(hyp_path))
            raw.set_annotations(annot, emit_warning=False)
            
            # 转换标签并切成 30s
            events, event_id = mne.events_from_annotations(
                raw, 
                event_id=self.stage_mapping, 
                chunk_duration=self.epoch_len_sec,
                verbose=False
            )
            
            # 切片 Epoching
            tmax = self.epoch_len_sec - (1.0 / raw.info['sfreq'])
            epochs = mne.Epochs(
                raw, 
                events=events, 
                event_id=event_id, 
                tmin=0, 
                tmax=tmax, 
                baseline=None,
                preload=True,
                verbose=False
            )

            # ================= [Stage 4] 写入 HDF5 =================
            clean_data = epochs.get_data(copy=True) 
            labels = epochs.events[:, 2]
            
            if len(labels) == 0:
                print(f"⚠️ {subject_id} 没有提取到有效片段")
                return None

            # 单位转换: V -> µV
            clean_data = clean_data * 1e6

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"sub_{subject_id}.h5"

            subject_attrs = SubjectAttrs(
                subject_id=subject_id,
                dataset_name="SleepEDF2018",
                task_type="sleep_staging",
                downstream_task_type="classification",
                rsFreq=self.target_fs,
                chn_name=available_chs,
                num_labels=5,
                category_list=['W', 'N1', 'N2', 'N3', 'R'],
                chn_type="EEG",
                montage="standard_1020"
            )

            with HDF5Writer(str(output_path), subject_attrs) as writer:
                trial_attrs = TrialAttrs(trial_id=1, session_id=1)
                trial_name = writer.add_trial(trial_attrs)
                
                for i in range(len(labels)):
                    seg_data = clean_data[i]
                    lbl = labels[i]
                    
                    segment_attrs = SegmentAttrs(
                        segment_id=i,
                        start_time=i * self.epoch_len_sec,
                        end_time=(i+1) * self.epoch_len_sec,
                        time_length=self.epoch_len_sec,
                        label=np.array([lbl]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)

            return str(output_path)

        except Exception as e:
            print(f"❌ 处理错误 {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return None