import pickle
import numpy as np
import os
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

data_base = f'/mnt/dataset4/daily_eeg_emotion/Data/CIRE'
out_base = f'/mnt/dataset2/Processed_datasets/EEG_Bench/CIRE'

target_sfreq = 200
sample_T = 2.0 # EEG sample window timelength in s
sample_stride = 2.0 # # EEG sample window stride in s
sample_samples = int(sample_T * target_sfreq)
stride_samples = int(sample_stride * target_sfreq)

MISSING_SESSIONS = [
    (5, 2),  # sub-05 ses-02
    (32, 5)  # sub-32 ses-05
]

data_segments = []
for sub_id in range(1, 39):
    data_sub = []
    sub_str = f'sub-{sub_id:02d}'
    
    sub_h5_path = os.path.join(out_base, f'{sub_str}.h5')
    os.makedirs(out_base, exist_ok=True)
    with h5py.File(sub_h5_path, 'w') as f:
        print(sub_str)
        print(f"\n处理 {sub_str}")
        for ses in range(1, 6):
            if (sub_id, ses) in MISSING_SESSIONS:
                print(f"Skipping known missing session: {sub_str} ses-{ses:02d}")
                continue
            
            pkl_path = os.path.join(data_base, sub_str, f'ses-{ses:02d}', f'{sub_str}_ses-{ses:02d}_trial_data.pkl')
            
            if not os.path.exists(pkl_path):
                print(f"Warning: file not found {pkl_path}")
                continue

            try:
                with open(pkl_path, 'rb') as file:
                    trial_dict = pickle.load(file)
            except Exception as e:
                print(f"Error: Failed to read file {pkl_path}: {e}")
                continue
            
            eeg_data = trial_dict['data']    # (n_trials, n_channels, T)
            labels = trial_dict['labels']
            fs = trial_dict['fs']
            ch_names = trial_dict['ch_names']
            ses_grp = f.create_group(f"session{ses}")
            
            for i_trial, data_trial in enumerate(eeg_data):
                if sub_id == 17 and ses == 1 and 48 <= i_trial <= 55:
                    print(f"Skipping corrupted trial: sub-{sub_id:02d} ses-{ses:02d} trial-{i_trial}")
                    continue
                print(f'sub {sub_id} trial {i_trial}: {data_trial.shape}')
                trial_grp = ses_grp.create_group(f"trial{i_trial}")
                n_samples = data_trial.shape[-1]
                sub_trial_label = labels[i_trial]
                n_slices = (n_samples - sample_samples + stride_samples) // stride_samples
                sub_trial_label = np.array(sub_trial_label).reshape(-1, 1)  # (1, 1)
                sub_trial_label = stretch_axis(sub_trial_label, n_slices)
                for i_slice, start in enumerate(range(0, n_samples - sample_samples + 1, stride_samples)):
                    end = start + sample_samples
                    slice_data = data_trial[:, start:end]
                    slice_grp = trial_grp.create_group(f"sample{i_slice}")
                    dset = slice_grp.create_dataset('eeg', data=slice_data)
                    dset.attrs['rsFreq'] = target_sfreq
                    dset.attrs['label'] = sub_trial_label[:, i_slice]
                    dset.attrs['subject_id'] = sub_id
                    dset.attrs['trial_id'] = i_trial
                    dset.attrs['session_id'] = ses
                    dset.attrs['segment_id'] = i_slice
                    dset.attrs['time_length'] = sample_T
                    dset.attrs['dataset_name'] = 'CIRE'
                    dset.attrs['chn_name'] = ch_names
                    dset.attrs['chn_pos'] = 'None'
                    dset.attrs['chn_ori'] = 'None'
                    dset.attrs['chn_type'] = 'EEG'
                    dset.attrs['montage'] = '10_20'