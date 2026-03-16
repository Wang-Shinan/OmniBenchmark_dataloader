# SEED-FRA/GER 数据集处理指南

## 当前状态

由于 ANT Neuro 格式的 .cnt 文件过大（~961MB），`antio` 库在处理时会发生段错误。**需要先将文件转换为其他格式**。

## 解决方案

### 方案1：使用 MATLAB/EEGLAB 转换（推荐）

1. **安装 EEGLAB**
   ```bash
   # 下载 EEGLAB: https://sccn.ucsd.edu/eeglab/download.php
   ```

2. **在 MATLAB 中运行转换脚本**
   ```matlab
   % 添加 EEGLAB 到路径
   addpath('/path/to/eeglab');
   
   % 转换 SEED-FRA 文件
   cnt_dir = '/mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw';
   output_dir = '/mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw';
   files = dir(fullfile(cnt_dir, '*.cnt'));
   for i = 1:length(files)
       convert_ant_to_mat(fullfile(cnt_dir, files(i).name), output_dir);
   end
   
   % 转换 SEED-GER 文件
   cnt_dir = '/mnt/dataset2/Datasets/SEED_GER/German/01-EEG-raw';
   output_dir = '/mnt/dataset2/Datasets/SEED_GER/German/01-EEG-raw';
   files = dir(fullfile(cnt_dir, '*.cnt'));
   for i = 1:length(files)
       convert_ant_to_mat(fullfile(cnt_dir, files(i).name), output_dir);
   end
   ```

3. **运行处理脚本**
   ```bash
   # SEED-FRA
   cd /mnt/dataset2/benchmark_dataloader/datasets
   source /home/wangshinan/miniconda3/etc/profile.d/conda.sh
   conda activate hdf
   python run_dataset.py seed_fra \
       /mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw \
       --feature_dir /mnt/dataset2/Datasets/SEED_FRA/French/02-EEG-DE-features \
       --output_dir /mnt/dataset2/hdf5_datasets \
       --subjects 1
   
   # SEED-GER
   python run_dataset.py seed_ger \
       /mnt/dataset2/Datasets/SEED_GER/German/01-EEG-raw \
       --feature_dir /mnt/dataset2/Datasets/SEED_GER/German/02-EEG-DE-features \
       --output_dir /mnt/dataset2/hdf5_datasets \
       --subjects 1
   ```

### 方案2：等待 antio 库更新

如果未来 antio 库修复了段错误问题，可以直接使用 .cnt 文件。

## 文件格式支持

代码现在支持：
- ✅ **转换后的 .mat 文件**（优先使用）
- ⚠️ **原始 .cnt 文件**（如果 antio 库修复后可用）

代码会自动检测文件格式：
1. 优先查找 `.mat` 文件
2. 如果没有找到，查找 `.cnt` 文件
3. 对于 `.cnt` 文件，自动检测是 ANT Neuro 还是 Neuroscan 格式

## 注意事项

- 转换后的 .mat 文件会保存在原始目录中（与 .cnt 文件同一目录）
- 转换后的文件名格式：`{subject_id}_{session_id}.mat`
- 确保有足够的磁盘空间（每个 .mat 文件约 1-2GB）

