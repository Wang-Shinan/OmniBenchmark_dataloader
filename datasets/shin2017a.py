import numpy as np
import scipy.io
import mne
from pathlib import Path
from builder import EEGDatasetBuilder
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
from hdf5_io import HDF5Writer

class Shin2017Builder(EEGDatasetBuilder):
    def __init__(self, dataset_info, preproc_config, raw_data_dir):
        super().__init__(dataset_info, preproc_config, raw_data_dir)
        
        self.task_name = "dsr"
        
        self.target_fs = 200.0
        self.configs = {
            "nback": {"epoch": 4.0, "mode": "event"},
            "wg": {"epoch": 10.0, "mode": "event"},
            "dsr": {"epoch": 4.0, "mode": "sliding", "stride": 4.0}
        }
        
        if self.task_name not in self.configs:
            raise ValueError(f"Unknown task: {self.task_name}")
            
        cfg = self.configs[self.task_name]
        self.epoch_len_sec = cfg["epoch"]
        self.mode = cfg["mode"]
        self.stride_sec = cfg.get("stride", self.epoch_len_sec)

    def get_subject_ids(self) -> list:
        raw_dir = self.raw_data_dir
        sub_dirs = sorted(list(raw_dir.glob("VP*-EEG")))
        if not sub_dirs:
            sub_dirs = sorted(list(raw_dir.rglob("VP*-EEG")))
        ids = [d.name.split('-')[0] for d in sub_dirs]
        return sorted(list(set(ids)))

    def get_raw_file_path(self, subject_id) -> Path:
        return self.raw_data_dir

    def get_label_file_path(self, subject_id) -> Path:
        return self.raw_data_dir

    def get_trial_info(self, subject_id) -> list[dict]:
        return []

    def build_subject(self, subject_id) -> str:
        found_dirs = list(self.raw_data_dir.rglob(f"{subject_id}-EEG"))
        if not found_dirs:
            return None
        sub_folder = found_dirs[0]

        cnt_path = sub_folder / f"cnt_{self.task_name}.mat"
        mrk_path = sub_folder / f"mrk_{self.task_name}.mat"

        if not cnt_path.exists():
            return None

        try:
            mat_cnt = scipy.io.loadmat(str(cnt_path), squeeze_me=True, struct_as_record=False)
            cnt = mat_cnt[f'cnt_{self.task_name}']
            
            raw_data = cnt.x.T 
            ch_names = list(cnt.clab)
            original_fs = float(cnt.fs)

            if np.max(np.abs(raw_data)) > 1:
                raw_data = raw_data * 1e-6
            
            info = mne.create_info(ch_names=ch_names, sfreq=original_fs, ch_types='eeg')
            raw = mne.io.RawArray(raw_data, info)
            
            if original_fs != self.target_fs:
                raw.resample(self.target_fs)
            
            raw.filter(l_freq=self.preproc_config.filter_low, 
                       h_freq=self.preproc_config.filter_high,
                       verbose=False)
            raw.notch_filter(freqs=self.preproc_config.filter_notch, verbose=False)

            clean_data = raw.get_data() 

            mat_mrk = scipy.io.loadmat(str(mrk_path), squeeze_me=True, struct_as_record=False)
            mrk = mat_mrk[f'mrk_{self.task_name}']
            
            events_ms = mrk.time
            events_sample = (events_ms / 1000.0 * self.target_fs).astype(int)
            
            if hasattr(mrk.y, 'ndim') and mrk.y.ndim > 1:
                labels = np.argmax(mrk.y, axis=0)
            else:
                labels = mrk.y

            if hasattr(mrk, 'className'):
                class_names = list(mrk.className)
            else:
                class_names = [f"Class_{i}" for i in range(10)]

            epoch_len_pts = int(self.epoch_len_sec * self.target_fs)
            segments = []
            final_labels = []

            if self.mode == "sliding":
                stride_pts = int(self.stride_sec * self.target_fs)
                block_len_pts = int(60.0 * self.target_fs)

                for i, start_idx in enumerate(events_sample):
                    block_label = int(labels[i])
                    
                    if i < len(events_sample) - 1:
                        block_end = events_sample[i+1]
                    else:
                        block_end = min(start_idx + block_len_pts, clean_data.shape[1])
                    
                    current_ptr = start_idx
                    while current_ptr + epoch_len_pts <= block_end:
                        seg = clean_data[:, current_ptr : current_ptr + epoch_len_pts]
                        seg_uV = seg * 1e6 
                        segments.append(seg_uV)
                        final_labels.append(block_label)
                        current_ptr += stride_pts
            else:
                for i, start_idx in enumerate(events_sample):
                    end_idx = start_idx + epoch_len_pts
                    if end_idx > clean_data.shape[1]:
                        continue
                    
                    seg = clean_data[:, start_idx:end_idx]
                    seg_uV = seg * 1e6 
                    segments.append(seg_uV)
                    final_labels.append(int(labels[i]))

            if not segments:
                return None

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"sub_{subject_id}.h5"

            subject_attrs = SubjectAttrs(
                subject_id=subject_id,
                dataset_name=f"Shin2017A_{self.task_name.upper()}",
                task_type="cognitive_task" if self.task_name != "dsr" else "resting_state",
                downstream_task_type="classification",
                rsFreq=self.target_fs,
                chn_name=ch_names,
                num_labels=len(class_names),
                category_list=class_names,
                chn_type="EEG",
                montage="standard_1020"
            )

            with HDF5Writer(str(output_path), subject_attrs) as writer:
                for i, seg_data in enumerate(segments):
                    lbl = final_labels[i]
                    trial_attrs = TrialAttrs(trial_id=i+1, session_id=1)
                    trial_name = writer.add_trial(trial_attrs)
                    segment_attrs = SegmentAttrs(
                        segment_id=0,
                        start_time=i * (self.stride_sec if self.mode == "sliding" else self.epoch_len_sec),
                        end_time=i * (self.stride_sec if self.mode == "sliding" else self.epoch_len_sec) + self.epoch_len_sec,
                        time_length=self.epoch_len_sec,
                        label=np.array([lbl]),
                    )
                    writer.add_segment(trial_name, segment_attrs, seg_data)

            return str(output_path)

        except Exception:
            import traceback
            traceback.print_exc()
            return None