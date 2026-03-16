import pickle
import os
import numpy as np
import h5py

def stretch_axis(arr, new_t):
    n, t = arr.shape
    # 原有的时间点坐标（0 到 t-1）
    old_indices = np.linspace(0, t - 1, t)
    # 目标时间点坐标（0 到 t-1，但采样 new_t 个点）
    new_indices = np.linspace(0, t - 1, new_t)
    
    # 对每一行进行线性插值
    # np.apply_along_axis 可以简化循环
    resized_matrix = np.apply_along_axis(
        lambda row: np.interp(new_indices, old_indices, row), 
        axis=1, 
        arr=arr
    )
    
    return resized_matrix

# ======================= 配置 =======================
dataset_name = 'EEG-SvRec'
target_sfreq = 200
sample_T = 2.0 # EEG sample window timelength in s
sample_stride = 2.0 # # EEG sample window stride in s
sample_samples = int(sample_T * target_sfreq)
stride_samples = int(sample_stride * target_sfreq)

output_path_base = r"/mnt/dataset4/daily_eeg_emotion/Data/EEG-SVRec"
path = r"/mnt/dataset4/daily_eeg_emotion/Data/EEG-SVRec/processed_data"
beh_file_path = r"/mnt/dataset2/Datasets/EEG-SVRec/metadata"
label_names = ['video_tag','like','immersion', 'satisf', 'arousal', 'valance']
subs = [i for i in range(1, 31) if i not in [1, 2, 4]]
for sub in subs:
    sub_id = f"{sub:02d}"
    with open(os.path.join(path, sub_id, f"{sub_id}_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)
    data = raw_data['data']
    chn_names = raw_data['chn_names']
    session_ids = raw_data['session_ids']
    
    with open(f"{beh_file_path}/{sub_id}_behavior_MAES.json", 'r') as f:
        beh_data = json.load(f)
    labels = []
    for key, value in beh_data.items():
        labels.append([value[label_name] for label_name in label_names])
    
    sub_h5_path = os.path.join(output_path_base, f"sub_{sub}.h5")
    with h5py.File(sub_h5_path, 'w') as h5f:
        print(sub)
        
        for i_trial, data_trial in enumerate(data):
                session = session_ids[i_trial]
                print(f'sub {sub} trial {i_trial}: {data_trial.shape}')
                trial_grp = h5f.create_group(f"trial{i_trial}")
                n_samples = data_trial.shape[-1]
                sub_trial_label = labels[i_trial]
                sub_trial_label = np.array(sub_trial_label)
                n_slices = (n_samples - sample_samples + stride_samples) // stride_samples
                for i_slice, start in enumerate(range(0, n_samples - sample_samples + 1, stride_samples)):
                    end = start + sample_samples
                    slice_data = data_trial[:, start:end]
                    slice_grp = trial_grp.create_group(f"sample{i_slice}")
                    dset = slice_grp.create_dataset('eeg', data=slice_data)
                    dset.attrs['rsFreq'] = target_sfreq
                    dset.attrs['label'] = sub_trial_label
                    dset.attrs['subject_id'] = sub
                    dset.attrs['trial_id'] = i_trial
                    dset.attrs['session_id'] = session
                    dset.attrs['segment_id'] = i_slice
                    dset.attrs['time_length'] = sample_T
                    dset.attrs['dataset_name'] = 'EEG-SvRec'
                    dset.attrs['chn_name'] = chn_names
                    dset.attrs['chn_pos'] = 'None'
                    dset.attrs['chn_ori'] = 'None'
                    dset.attrs['chn_type'] = 'EEG'
                    dset.attrs['montage'] = '10_20'
    