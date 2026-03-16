import os
import scipy.io
import numpy as np
import mne
import h5py
from pathlib import Path

# dataset config
print(np.__version__)
# 1.24.4
print(mne.__version__)
# 0.23.0

def stretch_axis(arr, T_prime):
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[np.newaxis, :] # 变为 [1, k]
    
    T = arr.shape[-1]
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, T_prime)
    
    # 对每一行执行插值
    res = np.apply_along_axis(lambda y: np.interp(x_new, x_old, y), axis=-1, arr=arr)
    
    return res.flatten() if is_1d else res


# ======================= 配置 =======================
dataset_name = 'THU-realtime'
target_sfreq = 200
sample_T = 2.0 # EEG sample window timelength in s
sample_stride = 2.0 # # EEG sample window stride in s

input_dir = r'/mnt/dataset4/qingzhu/RTEmotion/data_pre_reref/broad_band'
output_dir = f"/mnt/dataset2/Processed_datasets/EEG_Bench/THU-realtime_emo"
# output_dir = f"/mnt/dataset4/daily_eeg_emotion/Data/EmoEEG-MC/EmoEEG-MC_hdf5_test"
sample_samples = int(sample_T * target_sfreq)
stride_samples = int(sample_stride * target_sfreq)

n_sub = 13

# labels
beh_path = r'/mnt/dataset4/qingzhu/RTEmotion/ratings/old'
meta_path = r'/mnt/dataset4/qingzhu/RTEmotion/ratings'

labels = []   # (trials, n_slices, 4)  每个slice包含4个标签：valence_vid, arousal_vid, valence_rt_slice, arousal_rt_slice
valence_vid = scipy.io.loadmat(os.path.join(beh_path, 'valence_vid.mat'))['valence_vid']
valence_vid = valence_vid.mean(axis=0)  # (20,)
arousal_vid = scipy.io.loadmat(os.path.join(beh_path, 'arousal_vid.mat'))['arousal_vid']
arousal_vid = arousal_vid.mean(axis=0)  # (20,)

sample_num = scipy.io.loadmat(os.path.join(meta_path, 'sample_num.mat'))['sample_num'][0]
print(sample_num.shape)  # (1, 20) 


for i in range(20):
    label_trial = []
    
    video_fs = scipy.io.loadmat(os.path.join(meta_path, 'video_fs.mat'))['fs']
    video_fs = video_fs[i][0]
    if i in [9,10,13,15,16,19]:
        video_fs = video_fs + 1  # 修正帧率
        
    valence_vid_trial = valence_vid[i]
    arousal_vid_trial = arousal_vid[i]
    valence_rt = scipy.io.loadmat(os.path.join(beh_path, 'valence_rt.mat'))['valence_mean'][0][i]
    arousal_rt = scipy.io.loadmat(os.path.join(beh_path, 'arousal_rt.mat'))['arousal_mean'][0][i] 
    
    n_seconds = sample_num[i][0][0] / video_fs
    n_slices = n_seconds / 2 # 每2秒一个片段
    n_slices = int(n_slices)
    
    for j in range(n_slices):
        label_slices = []
        # 取两秒的标签
        valence_rt_slice = valence_rt[video_fs * 2 * j : video_fs * 2 * (j + 1)]
        arousal_rt_slice = arousal_rt[video_fs * 2 * j : video_fs * 2 * (j + 1)]
        
        label_slices.append(valence_vid_trial)
        label_slices.append(arousal_vid_trial)
        label_slices.append(valence_rt_slice)
        label_slices.append(arousal_rt_slice)
        
        label_trial.append(label_slices)
        
    labels.append(label_trial)
    
# ======================= 主处理 =======================
ch_names = [
    'Oz', 'O1', 'O2', 'P3', 'Pz', 'P4', 'T5', 'C3', 'C4', 'T6', 'F3', 'Fz', 'F4', 'T3', 'T4'
]

output_path_base = Path(output_dir)
output_path_base.mkdir(parents=True, exist_ok=True)
subs = None
subs = []

for sub in range(n_sub):
    # if(subs is not None and sub not in subs):
    #     continue
    sub_str = f"{(sub+1):02d}"
    
    sub_h5_path = output_path_base / f"sub_{sub}.h5"
    with h5py.File(sub_h5_path, 'w') as f:
            print(f"\n处理 sub_{sub}")
            name = f's{sub_str}.mat'
            full_path = os.path.join(input_dir, name)
            mat_data = scipy.io.loadmat(full_path)
            mat_data = mat_data['data'][0]  # (20,) -> (15, T)
            
            for i_trial, data_trial in enumerate(mat_data):
                print(f'sub {sub} trial {i_trial}: {data_trial.shape}')
                trial_grp = f.create_group(f"trial{i_trial}")
                n_samples = data_trial.shape[-1]
                sub_trial_label = labels[i_trial]
                n_slices = (n_samples - sample_samples + stride_samples) // stride_samples
                for i_slice, start in enumerate(range(0, n_samples - sample_samples + 1, stride_samples)):
                    end = start + sample_samples
                    slice_data = data_trial[:, start:end]
                    i_slice_label = i_slice / 200
                    # 超出范围补NaN
                    if i_slice_label >= len(sub_trial_label):
                        sub_label_slice = [np.nan, np.nan, np.full((video_fs * 2,), np.nan), np.full((video_fs * 2,), np.nan)]
                    else:
                        sub_label_slice = sub_trial_label[int(i_slice_label)]
                    flat_label = np.hstack([
                        sub_label_slice[0], 
                        sub_label_slice[1], 
                        sub_label_slice[2].ravel(), 
                        sub_label_slice[3].ravel()
                    ]).astype(np.float64)
                    slice_grp = trial_grp.create_group(f"sample{i_slice}")
                    dset = slice_grp.create_dataset('eeg', data=slice_data)
                    dset.attrs['rsFreq'] = target_sfreq
                    dset.attrs['label'] = flat_label
                    dset.attrs['subject_id'] = sub
                    dset.attrs['trial_id'] = i_trial
                    dset.attrs['session_id'] = 1
                    dset.attrs['segment_id'] = i_slice
                    dset.attrs['time_length'] = sample_T
                    dset.attrs['dataset_name'] = 'THU-realtime'
                    dset.attrs['chn_name'] = ch_names
                    dset.attrs['chn_pos'] = 'None'
                    dset.attrs['chn_ori'] = 'None'
                    dset.attrs['chn_type'] = 'EEG'
                    dset.attrs['montage'] = '10_20'