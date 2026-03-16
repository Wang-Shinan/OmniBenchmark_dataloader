# SEED-FRA/GER ANT Neuro 文件读取问题

## 问题描述

SEED-FRA 和 SEED-GER 数据集的原始数据是 ANT Neuro 格式的 `.cnt` 文件（约 961MB），使用 `antio` 库读取时会发生段错误（segmentation fault）。

## 解决方案

### 方案1：使用 MATLAB/EEGLAB 转换文件格式（推荐）

1. **安装 EEGLAB**
   ```bash
   # 下载 EEGLAB: https://sccn.ucsd.edu/eeglab/download.php
   ```

2. **在 MATLAB 中运行转换脚本**
   ```matlab
   % 添加 EEGLAB 到路径
   addpath('/path/to/eeglab');
   
   % 转换单个文件
   convert_ant_to_mat('/path/to/1_1.cnt', '/path/to/output');
   
   % 或批量转换
   cnt_dir = '/mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw';
   output_dir = '/mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw-mat';
   files = dir(fullfile(cnt_dir, '*.cnt'));
   for i = 1:length(files)
       convert_ant_to_mat(fullfile(cnt_dir, files(i).name), output_dir);
   end
   ```

3. **修改处理脚本以支持 .mat 格式**
   - 需要更新 `seed_fra.py` 和 `seed_ger.py` 以支持读取转换后的 .mat 文件

### 方案2：使用 Python 的 pyedflib 或其他库

如果文件可以转换为 EDF 格式：

```python
# 使用 pyedflib 或其他工具转换
# 然后修改代码支持 EDF 格式
```

### 方案3：联系数据集提供者

询问是否有其他格式的数据文件（如 .edf、.fif、.mat）。

### 方案4：使用预处理后的特征文件（仅标签信息）

如果只需要标签信息，可以直接使用 `02-EEG-DE-features` 目录中的特征文件。但注意这些是特征数据，不是原始EEG数据。

## 当前状态

- ✅ 代码已更新以检测 ANT Neuro 格式
- ✅ 使用子进程隔离避免主进程崩溃
- ✅ 提供清晰的错误提示
- ❌ antio 库在处理大文件时仍会段错误

## 临时解决方案

如果需要立即处理数据，建议：

1. **使用 MATLAB/EEGLAB 批量转换文件**
2. **或等待 antio 库更新修复此问题**
3. **或使用其他EEG处理工具链**

## 相关链接

- [antio GitHub](https://github.com/ant-neuro/antio)
- [EEGLAB](https://sccn.ucsd.edu/eeglab/)
- [MNE-Python ANT Support](https://mne.tools/stable/generated/mne.io.read_raw_ant.html)

