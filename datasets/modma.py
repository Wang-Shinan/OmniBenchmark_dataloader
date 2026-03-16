"""
Dataset Builder for MODMA (Resting State).
Path: datasets/modma.py
"""
import pandas as pd
import numpy as np
import mne
import scipy.io as sio
from pathlib import Path
from builder import EEGDatasetBuilder

class MODMABuilder(EEGDatasetBuilder):
    """
    MODMA Dataset (Resting State).
    Labels: 1 = Healthy Control, 2 = MDD
    """
    
    LABEL_MAP = {
        "HC": 1, "Healthy": 1, "Control": 1,
        "MDD": 2, "Depression": 2, "Patient": 2
    }

    def __init__(self, raw_data_dir, output_dir, dataset_info, preproc_config):
        preproc_config.output_dir = output_dir
        super().__init__(raw_data_dir=raw_data_dir, dataset_info=dataset_info, preproc_config=preproc_config)
        self.output_dir = Path(output_dir)
        self.label_dict = None 
        self.file_map = {}
        self.max_amplitude_uv = 600.0

    def load_meta_data(self):
        if self.label_dict is not None: return
        xlsx_files = list(self.raw_data_dir.glob("**/subjects_information_EEG_128channels_resting_lanzhou_2015*.xlsx"))
        if not xlsx_files: return
        
        target_xlsx = xlsx_files[0]
        for f in xlsx_files:
            if "clean" in f.name: target_xlsx = f
        
        try:
            df = pd.read_excel(target_xlsx, engine='openpyxl')
            df.columns = [str(c).strip().lower() for c in df.columns]
            self.label_dict = {}
            id_col = 'subject id' if 'subject id' in df.columns else df.columns[0]
            type_col = 'type' if 'type' in df.columns else df.columns[1]

            for _, row in df.iterrows():
                try:
                    sid = str(int(row[id_col])) 
                    raw_type = str(row[type_col]).strip()
                    if raw_type in self.LABEL_MAP:
                        self.label_dict[sid] = self.LABEL_MAP[raw_type]
                except: continue
        except Exception as e:
            print(f"❌ Failed to load Excel: {e}")

    def get_subject_ids(self) -> list:
        # 只扫描 Resting 文件夹
        files = sorted(list(self.raw_data_dir.glob("**/EEG_128channels_resting_lanzhou_2015/**/*.mat")))
        ids = []
        for f in files:
            try:
                ids.append(str(int(f.stem[:8])))
                self.file_map[str(int(f.stem[:8]))] = f
            except: continue
        return ids

    def get_raw_file_path(self, subject_id) -> Path:
        if not self.file_map: self.get_subject_ids()
        return self.file_map.get(str(subject_id))

    def get_trial_info(self, subject_id) -> list[dict]:
        self.load_meta_data()
        label = self.label_dict.get(str(subject_id), 0)
        file_path = self.get_raw_file_path(subject_id)
        duration = 60
        try:
            mat = sio.loadmat(str(file_path), variable_names=['samplingRate'])
            sfreq = mat['samplingRate'][0][0]
            whos = sio.whosmat(str(file_path))
            for name, shape, dtype in whos:
                if shape[0] in [128, 129]:
                    duration = shape[1] / sfreq
                    break
        except: pass
        return [{'trial_id': 0, 'session_id': 1, 'start_sec': 0, 'end_sec': duration, 'label': label}]

    def build_subject(self, subject_id: int) -> str:
        from hdf5_io import HDF5Writer
        from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
        
        file_path = self.get_raw_file_path(subject_id)
        if not file_path: return None
        try: mat = sio.loadmat(str(file_path))
        except: return None

        data_array = None
        sfreq = 250
        if 'samplingRate' in mat: sfreq = mat['samplingRate'][0][0]
        for key in mat:
            if not key.startswith('__') and isinstance(mat[key], np.ndarray):
                if mat[key].shape[0] in [128, 129]:
                    data_array = mat[key]
                    break
        if data_array is None: return None
        if data_array.shape[0] == 129: data_array = data_array[:128, :]

        ch_names = [f"E{i+1}" for i in range(data_array.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data_array, info, verbose=False)
        
        if self.preproc_config.filter_notch > 0:
            try: raw.notch_filter(self.preproc_config.filter_notch, verbose=False)
            except: pass
        raw.filter(self.preproc_config.filter_low, self.preproc_config.filter_high, verbose=False)
        raw.resample(self.preproc_config.target_sfreq, verbose=False)
        
        processed_data = raw.get_data()
        if np.abs(processed_data).max() < 1e-3:
            processed_data = processed_data * 1e6

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"
        trial_infos = self.get_trial_info(subject_id)
        
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name="MODMA",
            task_type="resting_state",
            downstream_task_type="classification",
            rsFreq=self.preproc_config.target_sfreq,
            chn_name=ch_names,
            num_labels=2,
            category_list=["Healthy Controls", "Major Depressive Disorder"],
            chn_pos=None, chn_ori=None, chn_type="EEG", montage="HydroCel GSN 128"
        )

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial_info in trial_infos:
                trial_attrs = TrialAttrs(trial_id=trial_info['trial_id'], session_id=trial_info['session_id'])
                trial_name = writer.add_trial(trial_attrs)
                
                segments = self.segment_trial(processed_data, trial_info['start_sec'], trial_info['end_sec'])
                for seg_id, (seg_data, duration) in enumerate(segments):
                    # 简单校验
                    if np.abs(seg_data).max() > self.max_amplitude_uv: continue
                    
                    segment_attrs = SegmentAttrs(
                        segment_id=seg_id,
                        start_time=trial_info['start_sec'] + seg_id * self.preproc_config.stride_sec,
                        end_time=trial_info['start_sec'] + seg_id * self.preproc_config.stride_sec + duration,
                        time_length=duration,
                        label=np.array([trial_info['label']])
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)
        return str(output_path)