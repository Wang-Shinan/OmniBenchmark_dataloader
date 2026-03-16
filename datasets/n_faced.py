import csv
import os
import numpy as np
import h5py
import pickle
import mne

# mapping
# label in total: 9 + 6 + 6 + 6 = 27
label_type1 = [0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*4 + [5]*3 + [6]*3 + [7]*3 + [8]*3
label_type2 = ['neg_a_1', 'neg_a_2', 'neg_a_3',
               'neg_d_1', 'neg_d_2', 'neg_d_3',
               'neg_f_1', 'neg_f_2', 'neg_f_3',
               'neg_s_1', 'neg_s_2', 'neg_s_3',
               'neu_1', 'neu_2', 'neu_3', 'neu_4',
               'pos_a_1', 'pos_a_2', 'pos_a_3',
               'pos_i_1', 'pos_i_2', 'pos_i_3',
               'pos_j_1', 'pos_j_2', 'pos_j_3',
               'pos_t_1', 'pos_t_2', 'pos_t_3'
               ]

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

path = r'/mnt/dataset4/sitian/post_training/danmu/FACED/results/smooth_scores'
data_segments = []

target_sfreq = 200
sample_T = 2.0 # EEG sample window timelength in s
sample_stride = 2.0 # # EEG sample window stride in s
sample_samples = int(sample_T * target_sfreq)
stride_samples = int(sample_stride * target_sfreq)

output_path_base = r'/mnt/dataset2/Processed_datasets/EEG_Bench/n_FACED'
file_path = r'/mnt/dataset2/Datasets/FACED/Data/sub000/data.bdf'
raw = mne.io.read_raw_bdf(file_path, preload=False)

for i in range(123):
    sub_h5_path = os.path.join(output_path_base, f"sub_{i:03d}.h5")
    os.makedirs(os.path.dirname(sub_h5_path), exist_ok=True)
    with h5py.File(sub_h5_path, 'w') as file_final:
        
        # label: (trial, label_dim, 1)
        label_dim1 = 9
        label_dim2 = 18
        
        trial_labels1 = np.zeros((28, label_dim1))
        
        trial_segments = []
        sub = f'sub{i:03d}'
        # test_path = r'/mnt/dataset4/daily_eeg_emotion/Data/FACED/Processed_Trials/sub003_processed.pkl'
        data_path = f'/mnt/dataset4/daily_eeg_emotion/Data/FACED/Processed_Trials/{sub}_processed.pkl'
        data_struct = pickle.load(open(data_path, 'rb'))
        full_data = data_struct['data']
        
        # 对每个被试的full_data做z-score标准化
        mean = np.mean(full_data, axis=1, keepdims=True)
        std = np.std(full_data, axis=1, keepdims=True)
        full_data = (full_data - mean) / std
        
        start_samples = data_struct['start_samples']
        # start_samples.sort()
        # # print(start_samples)
        end_samples = data_struct['end_samples']
        # end_samples.sort()
        video_label = data_struct['labels']
        original_order = data_struct['original_order']
        ch_names = raw.info['ch_names'][:32]
        
        mode1_rows = np.full((28, 100, 7), np.nan)
        mode2_rows = np.full((28, 100, 7), np.nan)
        mode3_rows = np.full((28, 100, 7), np.nan)
        
        for j in range(28):
            # print(f'Processing {sub} trial {j}...')
            s = int(start_samples[j])
            
            # 对于sub036-sub060，去掉开头7s
            if 36 <= i <= 60:
                s += 7 * target_sfreq
                
            e = int(end_samples[j])
            if np.isnan(s) or np.isnan(e):
                print(f'sub {sub} trial {j}: missing data, skipped.')
                continue
            segment = full_data[:, s:e]
            trial_segments.append(segment)
            video_idx = j
            
            # one-hot
            label1 = label_type1[video_idx]
            trial_labels1[j, label1] = 1
            
            # 读取label_type2对应的标签
            mode1_name = os.path.join(path, f'{label_type2[video_idx]}_smooth_mode1.csv')
            mode2_name = os.path.join(path, f'{label_type2[video_idx]}_smooth_mode2.csv')
            mode3_name = os.path.join(path, f'{label_type2[video_idx]}_smooth_mode3.csv')
            
            # 判断文件是否存在
            if os.path.exists(mode1_name):
                with open(mode1_name, 'r') as f:
                    reader = csv.reader(f)
                    mode1_rows_tr = [row for row in reader]
                    for k in range(len(mode1_rows_tr[1:])):
                        for l in range(7):
                            mode1_rows[j][k][l] = float(mode1_rows_tr[k+1][l])
            if os.path.exists(mode2_name):
                with open(mode2_name, 'r') as f:
                    reader = csv.reader(f)
                    mode2_rows_tr = [row for row in reader]
                    for k in range(len(mode2_rows_tr[1:])):
                        for l in range(7):
                            mode2_rows[j][k][l] = float(mode2_rows_tr[k+1][l])
            if os.path.exists(mode3_name):
                with open(mode3_name, 'r') as f:
                    reader = csv.reader(f)
                    mode3_rows_tr = [row for row in reader]
                    for k in range(len(mode3_rows_tr[1:])):
                        for l in range(7):
                            mode3_rows[j][k][l] = float(mode3_rows_tr[k+1][l])

        trial_labels1 = trial_labels1.reshape((28, label_dim1, 1))
        trial_labels1 = np.array(trial_labels1)
        # print(f'{sub}has segments:{len(trial_segments)}')
        # print(f'{sub}trial_labels1:{trial_labels1.shape}')
        
        for i_trial, data_trial in enumerate(trial_segments):
                # print(f'{sub} trial {i_trial}: {data_trial.shape}')
                trial_grp = file_final.create_group(f"trial{i_trial}")
                n_samples = data_trial.shape[-1]
                sub_trial_label1 = trial_labels1[i_trial]
                n_slices = (n_samples - sample_samples + stride_samples) // stride_samples
                sub_trial_label1 = stretch_axis(sub_trial_label1, n_slices) 
                trial_labels2 = np.full(label_dim2, np.nan)
                for i_slice, start in enumerate(range(0, n_samples - sample_samples + 1, stride_samples)):
                    end = start + sample_samples
                    slice_data = data_trial[:, start:end]
                    slice_grp = trial_grp.create_group(f"sample{i_slice}")
                    
                    for label2_dim in range(6):
                        trial_labels2[label2_dim] = mode1_rows[i_trial][i_slice][label2_dim + 1]
                        trial_labels2[label2_dim + 6] = mode2_rows[i_trial][i_slice][label2_dim + 1]
                        trial_labels2[label2_dim + 12] = mode3_rows[i_trial][i_slice][label2_dim + 1]

                    trial_labels1_con = sub_trial_label1[:, i_slice]
                    # print(sub_trial_label1.shape)
                    # print(trial_labels2.shape)
                    sub_trial_label = np.concatenate((trial_labels1_con, trial_labels2), axis=0)

                    dset = slice_grp.create_dataset('eeg', data=slice_data)
                    dset.attrs['rsFreq'] = target_sfreq
                    dset.attrs['label'] = sub_trial_label
                    dset.attrs['subject_id'] = sub
                    dset.attrs['trial_id'] = i_trial
                    dset.attrs['session_id'] = 0
                    dset.attrs['segment_id'] = i_slice
                    dset.attrs['time_length'] = sample_T
                    dset.attrs['dataset_name'] = 'FACED'
                    dset.attrs['chn_name'] = ch_names
                    dset.attrs['chn_pos'] = 'None'
                    dset.attrs['chn_ori'] = 'None'
                    dset.attrs['chn_type'] = 'EEG'
                    dset.attrs['montage'] = '10_20'
                    
    print(f'Subject {sub} processing completed.')
    with h5py.File(sub_h5_path, 'r') as file_check:
        for n_trial in range(28):
            for n_segment in range(len(file_check[f'trial{n_trial}'])):
                eeg_data = file_check[f'trial{n_trial}'][f'sample{n_segment}']['eeg'][:]
                label = file_check[f'trial{n_trial}'][f'sample{n_segment}']['eeg'].attrs['label']
                # 如果label的后18维全是nan，则说明该段数据的情绪标签缺失
                if np.isnan(label[9:]).all():
                    print(f'Trial {n_trial} Segment {n_segment} has missing emotion labels.')
        