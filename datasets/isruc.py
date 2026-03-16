import numpy as np
import scipy.io
import mne
from pathlib import Path
from builder import EEGDatasetBuilder
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
from hdf5_io import HDF5Writer

class ISRUCBuilder(EEGDatasetBuilder):
    def __init__(self, dataset_info, preproc_config, raw_data_dir):
        super().__init__(dataset_info, preproc_config, raw_data_dir)
        self.category_list = ['Wake', 'N1', 'N2', 'N3', 'REM']
        self.label_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 3,
            5: 4
        }

    def get_subject_ids(self) -> list:
        files = sorted(self.raw_data_dir.glob("subject*.mat"))
        ids = [f.stem.replace('subject', '') for f in files]
        return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)

    def get_raw_file_path(self, subject_id) -> Path:
        return self.raw_data_dir / f"subject{subject_id}.mat"

    def get_label_file_path(self, subject_id) -> Path:
        root_dir = self.raw_data_dir.parent.parent 
        
        candidate_folders = [
            "ISURC_3",
            "ISURC_2",
            "ISURC_1",
            "ISRUC_S3",
            "ISRUC_S2",
            "ISRUC_S1"
        ]
        
        for name in candidate_folders:
            folder_path = root_dir / name
            if not folder_path.exists():
                continue

            s3_path = folder_path / subject_id / "1" / f"{subject_id}_1.txt"
            if s3_path.exists():
                return s3_path

            s2_path = folder_path / "1" / subject_id / f"{subject_id}_1.txt"
            if s2_path.exists():
                return s2_path

            normal_path = folder_path / subject_id / f"{subject_id}_1.txt"
            if normal_path.exists():
                return normal_path

            raw_path = folder_path / "RawData" / subject_id / f"{subject_id}_1.txt"
            if raw_path.exists():
                return raw_path
        
        print(f"⚠️ 无法定位 Subject {subject_id} 的标签，已搜索: {candidate_folders}")
        return Path(f"MISSING_LABEL_{subject_id}")

    def get_trial_info(self, subject_id) -> list[dict]:
        return []

    def build_subject(self, subject_id) -> str:
        mat_path = self.get_raw_file_path(subject_id)
        label_path = self.get_label_file_path(subject_id)

        if not mat_path.exists():
            print(f"❌ 信号文件缺失: {mat_path}")
            return None
        
        if not label_path.exists():
            print(f"❌ 标签文件缺失: {label_path}")
            print(f"   (程序在尝试寻找: {label_path})")
            return None

        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            
            with open(label_path, 'r') as f:
                raw_labels = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
                
        except Exception as e:
            print(f"❌ 读取错误 ID={subject_id}: {e}")
            return None

        target_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
        
        epochs_data_list = []
        found_channels = []

        for ch in target_channels:
            if ch in mat:
                ch_data = mat[ch]
                epochs_data_list.append(ch_data)
                found_channels.append(ch)
            else:
                print(f"⚠️ 警告: ID={subject_id} 缺失通道 {ch}")

        if not epochs_data_list:
            print(f"❌ 错误: 未找到任何有效EEG通道")
            return None

        stack_data = np.stack(epochs_data_list)
        mne_data = stack_data.transpose(1, 0, 2)
        
        n_epochs, n_channels, n_samples = mne_data.shape
        
        if len(raw_labels) != n_epochs:
            min_len = min(len(raw_labels), n_epochs)
            mne_data = mne_data[:min_len]
            raw_labels = raw_labels[:min_len]
            n_epochs = min_len

        fs = 200.0
        info = mne.create_info(ch_names=found_channels, sfreq=fs, ch_types='eeg')
        
        mne_data = mne_data * 1e-6
        
        epochs = mne.EpochsArray(mne_data, info, verbose=False)
        epochs.filter(l_freq=self.preproc_config.filter_low, 
                      h_freq=self.preproc_config.filter_high, 
                      verbose=False)
        
        processed_data = epochs.get_data() 
        processed_data = processed_data * 1e6

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="ISRUC-Sleep",
            task_type="sleep_staging",
            downstream_task_type="classification",
            rsFreq=self.preproc_config.target_sfreq,
            chn_name=found_channels,
            num_labels=len(self.category_list),
            category_list=self.category_list,
            chn_type="EEG",
            montage="standard_1020"
        )

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            epoch_sec = 30.0
            
            for i in range(n_epochs):
                label_code = raw_labels[i]
                
                if label_code not in self.label_map:
                    continue

                final_label = self.label_map[label_code]
                seg_data = processed_data[i] 

                trial_attrs = TrialAttrs(trial_id=i+1, session_id=1)
                trial_name = writer.add_trial(trial_attrs)
                
                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=i * epoch_sec,
                    end_time=(i+1) * epoch_sec,
                    time_length=epoch_sec,
                    label=np.array([final_label]),
                )
                
                writer.add_segment(trial_name, segment_attrs, seg_data)

        return str(output_path)