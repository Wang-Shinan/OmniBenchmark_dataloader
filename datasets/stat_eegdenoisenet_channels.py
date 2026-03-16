#!/usr/bin/env python3
"""
统计 EEGdenoiseNet 数据集的通道信息

用法:
    python stat_eegdenoisenet_channels.py /path/to/EEGdenoiseNet/data
"""

import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, cannot load .mat files")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available, cannot read HDF5 files")


def load_epochs_from_mat(mat_path: Path, epoch_type: str = "EEG"):
    """
    从 .mat 文件加载 epochs 数据
    
    Args:
        mat_path: .mat 文件路径
        epoch_type: epoch 类型描述（用于日志）
    
    Returns:
        numpy array 或 None（如果加载失败）
    """
    if not HAS_SCIPY:
        return None
    
    if not mat_path.exists():
        return None
    
    try:
        mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        try:
            mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        except Exception as e2:
            print(f"  Error loading {mat_path}: {e2}")
            return None
    
    # 查找数据键
    keys = [k for k in mat_data.keys() if not k.startswith("__")]
    
    if len(keys) == 0:
        print(f"  No data found in {mat_path}")
        return None
    
    # 尝试找到合适的数据键
    data_key = None
    for key in keys:
        key_lower = key.lower()
        if (epoch_type.lower() in key_lower or 
            'epoch' in key_lower or 
            'data' in key_lower or
            'signal' in key_lower):
            data_key = key
            break
    
    if data_key is None:
        data_key = keys[0]
        print(f"  Warning: Using first available key '{data_key}' from {keys}")
    
    epochs = mat_data[data_key]
    
    # 处理维度
    if isinstance(epochs, np.ndarray):
        # 移除单例维度
        while epochs.ndim > 1 and epochs.shape[0] == 1:
            epochs = epochs[0]
        
        # 处理不同形状
        if epochs.ndim == 1:
            epochs = epochs.reshape(1, -1)
        elif epochs.ndim == 2:
            # (n_epochs, n_samples) - 单通道
            pass
        elif epochs.ndim == 3:
            # (n_epochs, n_channels, n_samples) - 多通道
            pass
        else:
            print(f"  Unexpected shape: {epochs.shape}")
            return None
    else:
        print(f"  Data is not numpy array, type: {type(epochs)}")
        return None
    
    return epochs


def analyze_channels_in_hdf5(hdf5_dir: Path):
    """
    分析已处理的 HDF5 文件中的通道信息
    
    Args:
        hdf5_dir: HDF5 文件目录
    """
    if not HAS_H5PY:
        print("h5py not available, skipping HDF5 analysis")
        return
    
    h5_files = list(hdf5_dir.glob("sub_*.h5"))
    if not h5_files:
        print(f"No HDF5 files found in {hdf5_dir}")
        return
    
    print("\n" + "=" * 60)
    print("HDF5 文件通道统计")
    print("=" * 60)
    
    channel_counts = {}
    subject_channels = {}
    
    for h5_file in sorted(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                # 读取 subject 属性
                if 'chn_name' in f.attrs:
                    ch_names = f.attrs['chn_name']
                    if isinstance(ch_names, bytes):
                        ch_names = [ch_names]
                    elif isinstance(ch_names, (list, np.ndarray)):
                        pass
                    else:
                        ch_names = [str(ch_names)]
                    
                    n_channels = len(ch_names) if isinstance(ch_names, (list, np.ndarray)) else 1
                    
                    # 统计通道数
                    channel_counts[n_channels] = channel_counts.get(n_channels, 0) + 1
                    
                    # 记录每个 subject 的通道信息
                    subject_id = f.attrs.get('subject_id', h5_file.stem)
                    subject_channels[subject_id] = {
                        'n_channels': n_channels,
                        'ch_names': list(ch_names) if isinstance(ch_names, (list, np.ndarray)) else ch_names
                    }
                    
                    print(f"  {h5_file.name}: {n_channels} channels")
        except Exception as e:
            print(f"  Error reading {h5_file}: {e}")
    
    print(f"\n通道数统计:")
    for n_ch, count in sorted(channel_counts.items()):
        print(f"  {n_ch} 通道: {count} 个文件")
    
    # 检查通道名称一致性
    if subject_channels:
        all_ch_names = [info['ch_names'] for info in subject_channels.values()]
        if len(set(str(ch) for ch in all_ch_names)) == 1:
            print(f"\n✓ 所有文件的通道名称一致")
        else:
            print(f"\n⚠ 不同文件的通道名称不一致:")
            for subj_id, info in list(subject_channels.items())[:3]:
                print(f"  {subj_id}: {info['ch_names'][:5]}...")


def analyze_raw_epochs(data_dir: Path):
    """
    分析原始 epochs 数据文件的通道信息
    
    Args:
        data_dir: 数据目录（包含 EEG_all_epochs.mat, EOG_all_epochs.mat, EMG_all_epochs.mat）
    """
    print("\n" + "=" * 60)
    print("原始 Epochs 数据通道统计")
    print("=" * 60)
    
    # 查找 epochs 文件
    epochs_dir = data_dir / "data" if (data_dir / "data").exists() else data_dir
    
    eeg_file = epochs_dir / "EEG_all_epochs.mat"
    eog_file = epochs_dir / "EOG_all_epochs.mat"
    emg_file = epochs_dir / "EMG_all_epochs.mat"
    
    epoch_info = {}
    
    # 分析 EEG epochs
    if eeg_file.exists():
        print(f"\n分析 EEG epochs: {eeg_file}")
        eeg_epochs = load_epochs_from_mat(eeg_file, "EEG")
        if eeg_epochs is not None:
            if eeg_epochs.ndim == 2:
                n_epochs, n_samples = eeg_epochs.shape
                n_channels = 1
                print(f"  形状: ({n_epochs}, {n_samples}) - 单通道 epochs")
            elif eeg_epochs.ndim == 3:
                n_epochs, n_channels, n_samples = eeg_epochs.shape
                print(f"  形状: ({n_epochs}, {n_channels}, {n_samples}) - {n_channels} 通道 epochs")
            else:
                print(f"  未预期的形状: {eeg_epochs.shape}")
                n_channels = None
            
            epoch_info['EEG'] = {
                'n_channels': n_channels,
                'n_epochs': n_epochs,
                'shape': eeg_epochs.shape
            }
    else:
        print(f"\nEEG epochs 文件不存在: {eeg_file}")
    
    # 分析 EOG epochs
    if eog_file.exists():
        print(f"\n分析 EOG epochs: {eog_file}")
        eog_epochs = load_epochs_from_mat(eog_file, "EOG")
        if eog_epochs is not None:
            if eog_epochs.ndim == 2:
                n_epochs, n_samples = eog_epochs.shape
                n_channels = 1
                print(f"  形状: ({n_epochs}, {n_samples}) - 单通道 epochs")
            elif eog_epochs.ndim == 3:
                n_epochs, n_channels, n_samples = eog_epochs.shape
                print(f"  形状: ({n_epochs}, {n_channels}, {n_samples}) - {n_channels} 通道 epochs")
            else:
                print(f"  未预期的形状: {eog_epochs.shape}")
                n_channels = None
            
            epoch_info['EOG'] = {
                'n_channels': n_channels,
                'n_epochs': n_epochs,
                'shape': eog_epochs.shape
            }
    else:
        print(f"\nEOG epochs 文件不存在: {eog_file}")
    
    # 分析 EMG epochs
    if emg_file.exists():
        print(f"\n分析 EMG epochs: {emg_file}")
        emg_epochs = load_epochs_from_mat(emg_file, "EMG")
        if emg_epochs is not None:
            if emg_epochs.ndim == 2:
                n_epochs, n_samples = emg_epochs.shape
                n_channels = 1
                print(f"  形状: ({n_epochs}, {n_samples}) - 单通道 epochs")
            elif emg_epochs.ndim == 3:
                n_epochs, n_channels, n_samples = emg_epochs.shape
                print(f"  形状: ({n_epochs}, {n_channels}, {n_samples}) - {n_channels} 通道 epochs")
            else:
                print(f"  未预期的形状: {emg_epochs.shape}")
                n_channels = None
            
            epoch_info['EMG'] = {
                'n_channels': n_channels,
                'n_epochs': n_epochs,
                'shape': emg_epochs.shape
            }
    else:
        print(f"\nEMG epochs 文件不存在: {emg_file}")
    
    # 总结
    if epoch_info:
        print("\n" + "-" * 60)
        print("通道数总结:")
        print("-" * 60)
        
        channel_counts = {}
        for epoch_type, info in epoch_info.items():
            n_ch = info['n_channels']
            if n_ch is not None:
                key = f"{epoch_type} ({n_ch} channels)"
                channel_counts[key] = info['n_epochs']
                print(f"  {epoch_type}: {n_ch} 通道, {info['n_epochs']} 个 epochs")
        
        # 检查通道数是否一致
        unique_channels = set(info['n_channels'] for info in epoch_info.values() if info['n_channels'] is not None)
        if len(unique_channels) == 1:
            print(f"\n✓ 所有 epoch 类型的通道数一致: {unique_channels.pop()} 通道")
        elif len(unique_channels) > 1:
            print(f"\n⚠ 不同 epoch 类型的通道数不一致:")
            for epoch_type, info in epoch_info.items():
                if info['n_channels'] is not None:
                    print(f"  {epoch_type}: {info['n_channels']} 通道")
            print(f"\n建议: 使用 unify_channels=True 统一通道数")
            print(f"  最小通道数: {min(unique_channels)}")
            print(f"  最大通道数: {max(unique_channels)}")
            print(f"  最常见的通道数: {max(set(info['n_channels'] for info in epoch_info.values() if info['n_channels'] is not None), key=lambda x: sum(1 for info in epoch_info.values() if info['n_channels'] == x))}")
    else:
        print("\n未找到任何 epochs 数据文件")


def main():
    parser = argparse.ArgumentParser(
        description="统计 EEGdenoiseNet 数据集的通道信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析原始 epochs 数据
  python stat_eegdenoisenet_channels.py /path/to/EEGdenoiseNet/data
  
  # 分析已处理的 HDF5 文件
  python stat_eegdenoisenet_channels.py /path/to/hdf5/EEGdenoiseNet --hdf5_dir /path/to/hdf5/EEGdenoiseNet
        """
    )
    parser.add_argument("data_dir", help="数据目录（原始数据或 HDF5 输出目录）")
    parser.add_argument("--hdf5_dir", help="HDF5 文件目录（如果与 data_dir 不同）")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    hdf5_dir = Path(args.hdf5_dir) if args.hdf5_dir else data_dir
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return 1
    
    print("=" * 60)
    print("EEGdenoiseNet 数据集通道信息统计")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"HDF5 目录: {hdf5_dir}")
    
    # 分析原始 epochs 数据
    analyze_raw_epochs(data_dir)
    
    # 分析 HDF5 文件（如果存在）
    if hdf5_dir.exists() and any(hdf5_dir.glob("sub_*.h5")):
        analyze_channels_in_hdf5(hdf5_dir)
    
    print("\n" + "=" * 60)
    print("统计完成")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
