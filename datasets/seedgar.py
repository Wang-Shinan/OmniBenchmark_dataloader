"""
SEED-GER Dataset Builder.

SEED-GER Dataset: Emotion Recognition dataset.
- 8 subjects (ID: 1-8)
- 原始采样率：200 Hz
- 类别标签：negative, neutral, positive（3分类）
- 数据格式：CNT（ANT Neuro）
- 采集地区：欧洲（50Hz 陷波滤波）
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# 延迟导入 mne 与 scipy.io.loadmat，避免在模块导入阶段因 mne 内部副作用导致失败
HAS_MNE = False
mne = None
loadmat = None

def ensure_mne():
    """Ensure `mne` and `loadmat` are imported and available.

    Returns the imported `mne` module. Raises the original exception if import fails.
    """
    global HAS_MNE, mne, loadmat
    if HAS_MNE and mne is not None:
        return mne
    try:
        import mne as _mne
        from scipy.io import loadmat as _loadmat
        mne = _mne
        loadmat = _loadmat
        HAS_MNE = True
        return mne
    except Exception as e:
        HAS_MNE = False
        raise

try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

# -------------------------- 1. 对齐 README：数据集配置（完善 DatasetInfo） --------------------------
SEED_GER_INFO = DatasetInfo(
    dataset_name="SEED-GER_3class",  # 对齐 README 命名规范：数据集_类别数
    task_type=DatasetTaskType.EMOTION,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=3,  # 明确类别数
    category_list=["negative", "neutral", "positive"],  # 标签列表（索引对应标签值）
    sampling_rate=200.0,  # 原始采样率
    montage="10_10",  # 10-10 电极布局
    channels=[
        'FP1','FPZ','FP2',
        'AF3',    'AF4',
        'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
        'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
        'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
        'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
        'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
        'O1','OZ','O2'
    ],
)

# -------------------------- 2. 对齐 README：常量定义 --------------------------
# 需移除的通道（参考电极/EOG通道，当前无额外通道需移除）
SEED_GER_REMOVE_CHANNELS = []

# 默认振幅阈值（µV），对齐 README 情感任务推荐阈值 500 µV
DEFAULT_MAX_AMPLITUDE_UV = 500.0

# 标签元数据（固定映射，已完成 -1→0, 0→1, 1→2 的转换，对应 category_list 索引）
SEED_GER_LABEL_META = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, 0, -1, -1, 0, 1, 1]) + 1

# -------------------------- 3. 对齐 README：单位检测与转换函数 --------------------------
def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE (严格对齐 README 要求).

    Returns:
        tuple: (data_in_volts, detected_unit)
    """
    max_amp = np.abs(data).max()

    if max_amp > 1e-2:  # > 0.01，大概率是微伏（µV）
        return data / 1e6, "µV"
    elif max_amp > 1e-5:  # > 0.00001，大概率是毫伏（mV）
        return data / 1e3, "mV"
    else:  # 大概率已经是伏特（V）
        return data, "V"

# -------------------------- 4. 对齐 README：构建器类（完善所有要求方法） --------------------------
class SEEDGERBuilder:
    """Builder for SEED-GER dataset（完全对齐 README 指南要求）."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,  # 欧洲采集，50Hz 工频干扰
        file_format: str = "cnt",
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
    ):
        """
        初始化 SEED-GER 构建器（严格对齐 README 参数要求）。

        Args:
            raw_data_dir: 原始数据集文件目录
            output_dir: HDF5 文件输出目录
            target_sfreq: 目标采样率（SEED-GER 原始 200Hz，默认不重采样）
            window_sec: 窗口长度（秒）
            stride_sec: 步长长度（秒，与窗口等长为无重叠分段）
            filter_low: 带通滤波低频截止（对齐情感任务 0.1Hz）
            filter_high: 带通滤波高频截止（对齐情感任务 75Hz）
            filter_notch: 陷波滤波频率（欧洲 50Hz）
            file_format: 文件格式（仅支持 cnt）
            max_amplitude_uv: 试次振幅阈值（µV），超出则拒绝该试次
        """
        # 核心路径配置
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / "SEED-GER"  # 输出子目录以数据集命名
        
        # 数据处理参数
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 200.0  # SEED-GER 原始采样率固定 200Hz
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.file_format = file_format.lower()
        self.max_amplitude_uv = max_amplitude_uv

        # 计算采样点数量（对齐 README 窗口/步长采样点转换）
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        
        # 元数据加载（被试信息、试次时间戳）
        self._load_meta_info()

        # 验证统计量（对齐 README 试次验证要求）
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

    def _load_meta_info(self):
        """加载数据集元信息（被试、试次时间戳）"""
        # 被试元数据
        self.sub_meta = pd.DataFrame({
            'subject': [i for i in range(1, 9)],
            'sex': ['M', 'M', 'M', 'M', 'M', 'M', 'F', 'M'],
            'age': [20, 22, 26, 23, 21, 21, 24, 21]
        })

        # 试次时间戳（固定分段依据）
        self.trial_start_times = np.array([5.0, 411.0, 861.0, 1114.0, 1287.0, 1454.0, 
                                           1620.0, 1878.0, 2135.0, 2310.0, 2502.0, 2709.0, 
                                           3028.0, 3162.0, 3290.0, 3656.0, 3823.0, 4366.0])
        self.trial_end_times = np.array([136.0, 831.0, 1084.0, 1257.0, 1423.0, 1589.0, 
                                         1848.0, 2105.0, 2280.0, 2472.0, 2677.0, 2998.0, 
                                         3131.0, 3259.0, 3626.0, 3792.0, 4079.0, 4538.0])

        print(f"元数据加载完成：")
        print(f"  试次数量：{len(self.trial_start_times)}")
        print(f"  标签数量：{len(SEED_GER_LABEL_META)}")
        print(f"  被试数量：{len(self.sub_meta)}")

    # -------------------------- 对齐 README：获取被试 ID 列表 --------------------------
    def get_subject_ids(self) -> list[int]:
        """Get list of subject IDs (1-8)（严格对齐 README 方法要求）."""
        return list(range(1, 9))

    # -------------------------- 对齐 README：文件查找方法 --------------------------
    def _find_files(self, subject_id: int) -> dict[int, Path]:
        """
        Find all session files for a subject（返回 {session_id: file_path}）.
        适配原始数据命名规则：subject_session.cnt
        """
        if subject_id not in self.get_subject_ids():
            raise ValueError(f"无效被试 ID：{subject_id}，仅支持 1-8")
        
        files = {}
        ext = f".{self.file_format}"
        
        # 遍历 3 个会话（SEED-GER 最多 3 个会话 per 被试）
        for session_id in range(1, 4):
            file_path = self.raw_data_dir / f"{subject_id}_{session_id}{ext}"
            if file_path.exists():
                files[session_id] = file_path
        
        return files

    # -------------------------- 对齐 README：读取原始 CNT 文件 --------------------------
    def _read_raw_cnt(self, file_path: Path) -> mne.io.Raw:
        """Read CNT format file and convert to MNE Raw object（对齐 MNE 数据格式要求）."""
        ensure_mne()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_ant(
                str(file_path), 
                preload=True, 
                verbose=False,
            )
        
        # 通道名转为大写，保证与 DatasetInfo 配置一致
        raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})
        
        # 验证数据单位并转换为 Volts（MNE 内部标准单位）
        data = raw.get_data()
        data_volts, detected_unit = detect_unit_and_convert_to_volts(data)
        print(f"  检测到数据单位：{detected_unit}，已转换为 Volts 用于 MNE 处理")
        
        # 替换原始数据为转换后的 Volts 数据
        raw._data = data_volts
        
        # 打印文件基本信息
        print(f"  文件 {file_path.name} 信息：")
        print(f"    采样率：{raw.info['sfreq']:.2f} Hz")
        print(f"    采样点数：{raw.n_times}")
        print(f"    通道数：{len(raw.ch_names)}")
        
        return raw

    def _read_raw(self, file_path: Path) -> mne.io.Raw:
        """统一读取原始文件（仅支持 CNT 格式）."""
        if self.file_format != "cnt":
            raise ValueError(f"SEED-GER 仅支持 CNT 格式，当前指定：{self.file_format}")
        
        return self._read_raw_cnt(file_path)

    # -------------------------- 对齐 README：数据预处理方法 --------------------------
    def _preprocess(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing pipeline（严格对齐 README 预处理步骤）."""
        # 1. 移除指定通道（参考电极/无用通道）
        channels_to_drop = [ch for ch in SEED_GER_REMOVE_CHANNELS if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop, verbose=False)
            print(f"  已移除通道：{channels_to_drop}")
        
        # 2. 陷波滤波（50Hz，去除工频干扰）
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, fir_design='firwin', verbose=False)
            print(f"  已应用 {self.filter_notch} Hz 陷波滤波")
        
        # 3. 带通滤波（0.1-75Hz，情感任务标准频段）
        raw.filter(
            l_freq=self.filter_low, 
            h_freq=self.filter_high, 
            fir_design='firwin', 
            verbose=False
        )
        print(f"  已应用 {self.filter_low}-{self.filter_high} Hz 带通滤波")
        
        # 4. 重采样（仅当目标采样率与原始采样率不一致时）
        if abs(raw.info['sfreq'] - self.target_sfreq) > 1e-6:
            raw.resample(self.target_sfreq, verbose=False)
            print(f"  已重采样至 {self.target_sfreq} Hz")
        
        return raw

    # -------------------------- 对齐 README：试次振幅验证 --------------------------
    def _validate_trial(self, trial_data: np.ndarray) -> bool:
        """Check if trial amplitude is within acceptable range（µV）."""
        return np.abs(trial_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: int):
        """Report trial validation statistics（对齐 README 统计报告要求）."""
        valid_pct = (self.valid_trials / self.total_trials * 100) if self.total_trials > 0 else 0
        print(f"\n被试 {subject_id} 试次验证报告：")
        print(f"  振幅阈值：{self.max_amplitude_uv} µV")
        print(f"  总试次数：{self.total_trials}")
        print(f"  有效试次数：{self.valid_trials} ({valid_pct:.1f}%)")
        print(f"  被拒绝试次数：{self.rejected_trials} ({100-valid_pct:.1f}%)\n")

    # -------------------------- 对齐 README：提取试次数据 --------------------------
    def _extract_trials(self, raw: mne.io.Raw, session_id: int) -> list[dict]:
        """Extract trials from preprocessed raw data（基于固定时间戳分割）."""
        trials = []
        # 提取数据并转换为 µV（输出 HDF5 标准单位）
        data_volts = raw.get_data()
        data_uv = data_volts * 1e6  # Volts → µV
        total_duration = raw.n_times / self.target_sfreq

        # 遍历时间戳和标签
        for trial_id, (start_sec, end_sec, label) in enumerate(zip(
            self.trial_start_times, self.trial_end_times, SEED_GER_LABEL_META[:len(self.trial_start_times)]
        )):
            # 超出录制时长则跳过
            if end_sec > total_duration:
                warnings.warn(f"会话 {session_id} 的试次 {trial_id}（起始 {start_sec}s）超出录制时长，已跳过")
                continue

            # 转换为采样点索引
            start_idx = int(start_sec * self.target_sfreq)
            end_idx = int(end_sec * self.target_sfreq)
            trial_data = data_uv[:, start_idx:end_idx]

            # 整理试次信息
            trials.append({
                'data': trial_data,  # 已转换为 µV
                'label': int(label),  # 对应 category_list 索引
                'onset_time': start_sec,
                'duration': end_sec - start_sec,
                'trial_id': trial_id
            })

        return trials

    # -------------------------- 对齐 README：构建单个被试 HDF5 文件 --------------------------
    def build_subject(self, subject_id: int) -> str:
        """Build HDF5 file for a single subject（核心入口，完全对齐 README 要求）."""
        try:
            ensure_mne()
        except Exception as e:
            raise ImportError("需要安装 MNE 和 Scipy 以处理 SEED-GER 数据集：\n" + str(e)) from e

        # 重置验证统计量（针对当前被试）
        self.total_trials = 0
        self.valid_trials = 0
        self.rejected_trials = 0

        # 1. 查找当前被试的所有会话文件
        session_files = self._find_files(subject_id)
        if not session_files:
            raise FileNotFoundError(f"被试 {subject_id} 未找到任何有效 CNT 文件（命名规则：{subject_id}_session.cnt）")

        # 2. 初始化变量，存储所有有效试次
        all_trials = []
        ch_names = None
        global_trial_counter = 0

        # 3. 处理每个会话
        for session_id, file_path in session_files.items():
            print(f"\n正在处理：被试 {subject_id} → 会话 {session_id} → 文件 {file_path.name}")
            
            try:
                # 读取原始文件
                raw = self._read_raw(file_path)
                # 数据预处理
                raw = self._preprocess(raw)

                # 记录通道名称（仅首次获取）
                if ch_names is None:
                    ch_names = raw.ch_names

                # 提取试次数据
                trials = self._extract_trials(raw, session_id)
                
                # 整理试次数据，添加会话信息
                for trial in trials:
                    all_trials.append({
                        'data': trial['data'],
                        'label': trial['label'],
                        'session_id': session_id,
                        'trial_id': global_trial_counter,
                        'onset_time': trial['onset_time']
                    })
                    global_trial_counter += 1

            except Exception as e:
                print(f"【错误】处理被试 {subject_id} 会话 {session_id} 失败：{str(e)}")
                continue

        # 4. 验证试次数据是否有效
        if not all_trials:
            raise ValueError(f"被试 {subject_id} 未提取到有效试次数据")

        # 5. 构建被试属性（严格对齐 README SubjectAttrs 格式）
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=SEED_GER_INFO.dataset_name,
            task_type=SEED_GER_INFO.task_type.value,
            downstream_task_type=SEED_GER_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=SEED_GER_INFO.num_labels,  # 补充类别数
            category_list=SEED_GER_INFO.category_list,  # 补充标签列表
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=SEED_GER_INFO.montage,
        )

        # 6. 生成 HDF5 文件（对齐 README 分段写入逻辑）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for trial in all_trials:
                trial_data = trial['data']  # 已为 µV 格式
                self.total_trials += 1

                # 验证试次振幅，超出阈值则拒绝
                if not self._validate_trial(trial_data):
                    self.rejected_trials += 1
                    max_amp = np.abs(trial_data).max()
                    print(f"  跳过试次 {trial['trial_id']}：振幅 {max_amp:.1f} µV > {self.max_amplitude_uv} µV")
                    continue

                self.valid_trials += 1

                # 添加试次属性
                trial_attrs = TrialAttrs(
                    trial_id=trial['trial_id'],
                    session_id=trial['session_id'],
                )
                trial_name = writer.add_trial(trial_attrs)

                # 对齐 README：窗口分段（支持重叠/非重叠）
                n_samples = trial_data.shape[1]
                for i_slice, start in enumerate(range(0, n_samples - self.window_samples + 1, self.stride_samples)):
                    end = start + self.window_samples
                    slice_data = trial_data[:, start:end]

                    # 构建分段属性
                    segment_attrs = SegmentAttrs(
                        segment_id=i_slice,
                        start_time=(trial['onset_time'] + start / self.target_sfreq),
                        end_time=(trial['onset_time'] + end / self.target_sfreq),
                        time_length=self.window_sec,
                        label=np.array([trial['label']]),
                    )
                    writer.add_segment(trial_name, segment_attrs, slice_data)

        # 7. 输出验证统计报告
        self._report_validation_stats(subject_id)
        print(f"被试 {subject_id} 数据已保存至：{output_path}（有效试次 {self.valid_trials} 个）")
        return str(output_path)

    # -------------------------- 对齐 README：构建所有被试 --------------------------
    def build_all(self, subject_ids: list[int] = None) -> list[str]:
        """Build HDF5 files for all subjects（批量处理）."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
            except Exception as e:
                print(f"\n【错误】处理被试 {subject_id} 失败：{e}")
                continue

        return output_paths

# -------------------------- 5. 对齐 README：便捷构建函数 --------------------------
def build_seed_ger(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[int] = None,
    **kwargs,
) -> list[str]:
    """Convenience function to build SEED-GER dataset（对齐 README 便捷调用要求）."""
    builder = SEEDGERBuilder(raw_data_dir, output_dir, **kwargs)
    return builder.build_all(subject_ids)

# -------------------------- 6. 对齐 README：命令行运行入口 --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SEED-GER HDF5 dataset（适配 subject_session.cnt 命名规则）")
    parser.add_argument("raw_data_dir", help="原始 CNT 文件目录路径（文件命名：subject_session.cnt）")
    parser.add_argument("--output_dir", default="./hdf5", help="HDF5 文件输出目录")
    parser.add_argument("--subjects", nargs="+", type=int, help="指定要处理的被试 ID（1-8）")
    parser.add_argument("--window_sec", type=float, default=4.0, help="分段窗口长度（秒）")
    parser.add_argument("--stride_sec", type=float, default=4.0, help="分段步长（秒）")
    parser.add_argument("--max_amplitude_uv", type=float, default=500.0, help="试次振幅阈值（µV）")
    args = parser.parse_args()

    # 整理额外参数
    kwargs = {
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "max_amplitude_uv": args.max_amplitude_uv,
    }

    # 运行数据集构建
    build_seed_ger(args.raw_data_dir, args.output_dir, args.subjects, **kwargs)