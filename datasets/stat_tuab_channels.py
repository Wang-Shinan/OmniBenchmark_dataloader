"""
统计 TUAB 数据集中的通道数分布。

遍历所有 EDF 文件，统计每个文件的通道数和通道名称。
"""

from pathlib import Path
from collections import defaultdict, Counter
import warnings

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False
    print("Warning: pyedflib not available. Trying mne...")
    try:
        import mne
        HAS_MNE = True
        HAS_PYEDFLIB = False
    except ImportError:
        HAS_MNE = False
        print("Error: Neither pyedflib nor mne available. Please install one: pip install pyedflib or pip install mne")

def count_channels_in_file(edf_path: Path):
    """读取单个 EDF 文件并返回通道信息。"""
    # Try pyedflib first (faster, doesn't load full data)
    if HAS_PYEDFLIB:
        try:
            edf_file = pyedflib.EdfReader(str(edf_path))
            n_channels = edf_file.signals_in_file
            channel_names = [edf_file.getSignalLabel(i) for i in range(n_channels)]
            edf_file.close()
            return n_channels, channel_names
        except Exception as e:
            # Fallback to mne if pyedflib fails
            pass
    
    # Fallback to mne
    if HAS_MNE:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            
            n_channels = len(raw.ch_names)
            channel_names = raw.ch_names.copy()
            
            return n_channels, channel_names
        except Exception as e:
            pass
    
    return None, None

def stat_tuab_channels(raw_data_dir: str):
    """
    统计 TUAB 数据集中的通道数分布。
    
    Args:
        raw_data_dir: TUAB 数据集根目录（包含 train/eval 子目录）
    """
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        print(f"Error: Directory not found: {raw_data_dir}")
        return
    
    # 查找所有 EDF 文件
    edf_files = list(raw_data_path.rglob("*.edf"))
    
    if not edf_files:
        print(f"No EDF files found in {raw_data_dir}")
        return
    
    print(f"Found {len(edf_files)} EDF files")
    print("=" * 60)
    
    # 统计信息
    channel_count_dist = Counter()  # 通道数分布
    channel_names_by_count = defaultdict(set)  # 每个通道数对应的通道名称集合
    file_count_by_channels = defaultdict(int)  # 每个通道数对应的文件数
    error_files = []
    
    # 按 split 和 label 分组统计
    stats_by_split_label = defaultdict(lambda: defaultdict(Counter))
    
    # 处理每个文件
    for i, edf_file in enumerate(edf_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(edf_files)} files...")
        
        # 确定 split 和 label
        parts = edf_file.parts
        split = "unknown"
        if "train" in parts:
            split = "train"
        elif "eval" in parts:
            split = "eval"
        
        label = "unknown"
        if "normal" in parts:
            label = "normal"
        elif "abnormal" in parts:
            label = "abnormal"
        
        # 读取通道信息
        n_channels, channel_names = count_channels_in_file(edf_file)
        
        if n_channels is None:
            error_files.append(edf_file)
            continue
        
        # 更新统计信息
        channel_count_dist[n_channels] += 1
        file_count_by_channels[n_channels] += 1
        
        if channel_names:
            channel_names_by_count[n_channels].update(channel_names)
        
        # 按 split 和 label 统计
        stats_by_split_label[split][label][n_channels] += 1
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("总体统计")
    print("=" * 60)
    print(f"总文件数: {len(edf_files)}")
    print(f"成功读取: {len(edf_files) - len(error_files)}")
    print(f"读取失败: {len(error_files)}")
    
    print("\n通道数分布:")
    print("-" * 60)
    print(f"{'通道数':<10} {'文件数':<10} {'占比':<10}")
    print("-" * 60)
    
    total_valid = sum(channel_count_dist.values())
    for n_ch in sorted(channel_count_dist.keys()):
        count = channel_count_dist[n_ch]
        percentage = count / total_valid * 100 if total_valid > 0 else 0
        print(f"{n_ch:<10} {count:<10} {percentage:>6.2f}%")
    
    print("\n各通道数对应的通道名称示例:")
    print("-" * 60)
    for n_ch in sorted(channel_names_by_count.keys()):
        ch_names = sorted(list(channel_names_by_count[n_ch]))
        print(f"\n{n_ch} 通道 (共 {file_count_by_channels[n_ch]} 个文件):")
        # 显示前 30 个通道名
        if len(ch_names) <= 30:
            print(f"  所有通道: {ch_names}")
        else:
            print(f"  前 30 个通道: {ch_names[:30]}")
            print(f"  ... (共 {len(ch_names)} 个不同的通道名)")
    
    # 按 split 和 label 统计
    print("\n" + "=" * 60)
    print("按 Split 和 Label 统计")
    print("=" * 60)
    
    for split in sorted(stats_by_split_label.keys()):
        print(f"\n{split.upper()}:")
        for label in sorted(stats_by_split_label[split].keys()):
            print(f"  {label}:")
            dist = stats_by_split_label[split][label]
            total = sum(dist.values())
            for n_ch in sorted(dist.keys()):
                count = dist[n_ch]
                percentage = count / total * 100 if total > 0 else 0
                print(f"    {n_ch} 通道: {count} 个文件 ({percentage:.2f}%)")
    
    # 错误文件列表
    if error_files:
        print("\n" + "=" * 60)
        print(f"读取失败的文件 ({len(error_files)} 个):")
        print("=" * 60)
        for err_file in error_files[:20]:  # 只显示前 20 个
            print(f"  {err_file}")
        if len(error_files) > 20:
            print(f"  ... 还有 {len(error_files) - 20} 个文件")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="统计 TUAB 数据集中的通道数分布")
    parser.add_argument(
        "raw_data_dir",
        nargs="?",
        default="/mnt/dataset2/Datasets/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf",
        help="TUAB 数据集根目录（默认: /mnt/dataset2/Datasets/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf）"
    )
    
    args = parser.parse_args()
    
    stat_tuab_channels(args.raw_data_dir)
