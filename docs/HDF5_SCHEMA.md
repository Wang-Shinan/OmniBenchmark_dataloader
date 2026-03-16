# EEG-FM-Bench HDF5 数据结构规范

## 概述

本文档描述了 EEG-FM-Bench 基准测试中使用的 HDF5 数据格式规范。该格式采用层次化结构，支持多被试、多试验、多片段的 EEG 数据存储。

---

## HDF5 文件结构

每个被试对应一个 HDF5 文件：`sub_{subject_id}.h5`

```
sub_0.h5
│
├── 📋 Root Attributes (Subject-level)
│   ├── subject_id: int | str
│   ├── dataset_name: str              # e.g., "SEED_3class", "BCIC2A_4class"
│   ├── task_type: str                 # e.g., "emotion", "motor_imaginary"
│   ├── downstream_task_type: str       # e.g., "classification", "regression"
│   ├── rsFreq: float                  # Sampling frequency (Hz)
│   ├── chn_name: list[str]            # Channel names
│   ├── num_labels: int                # Number of classes (default: 0)
│   ├── category_list: list[str]       # Label names, index = label number
│   ├── chn_pos: np.ndarray | None     # Channel positions (n_channels, 3)
│   ├── chn_ori: np.ndarray | None     # Channel orientations (n_channels, 3)
│   ├── chn_type: str                  # e.g., "EEG"
│   └── montage: str                    # e.g., "10_20"
│
├── 📁 trial0 (Group)
│   ├── 📋 Attributes
│   │   ├── trial_id: int
│   │   └── session_id: int
│   │
│   ├── 📁 segment0 (Group)
│   │   └── 📊 eeg (Dataset: float32, shape=[n_channels, n_timepoints])
│   │       ├── 📋 Attributes
│   │       │   ├── segment_id: int
│   │       │   ├── start_time: float      # Start time in seconds
│   │       │   ├── end_time: float        # End time in seconds
│   │       │   ├── time_length: float     # Duration in seconds
│   │       │   └── label: np.ndarray      # Label array (shape varies)
│   │       └── 📈 Data: EEG signal (µV)
│   │
│   ├── 📁 segment1 (Group)
│   │   └── ...
│   │
│   └── 📁 segmentN (Group)
│       └── ...
│
├── 📁 trial1 (Group)
│   └── ...
│
└── 📁 trialM (Group)
    └── ...
```

---

## 数据结构层次

### 1. Subject Level (文件根属性)

**位置**: HDF5 文件根目录的 `attrs`

**用途**: 存储被试级别的元数据，所有 trial 和 segment 共享

| 字段 | 类型 | 必需 | 说明 | 示例 |
|------|------|------|------|------|
| `subject_id` | int \| str | ✅ | 被试唯一标识符 | `1` 或 `"sub_001"` |
| `dataset_name` | str | ✅ | 数据集名称（数据集+实验） | `"SEED_3class"`, `"BCIC2A_4class"` |
| `task_type` | str | ✅ | 任务类型 | `"emotion"`, `"motor_imaginary"` |
| `downstream_task_type` | str | ✅ | 下游任务类型 | `"classification"`, `"regression"` |
| `rsFreq` | float | ✅ | 采样频率 (Hz) | `250.0`, `128.0` |
| `chn_name` | list[str] | ✅ | 通道名称列表 | `["FZ", "FC3", "C3", ...]` |
| `num_labels` | int | ⚠️ | 类别数量 | `3`, `4` (默认: `0`) |
| `category_list` | list[str] | ⚠️ | 类别名称列表 | `["left", "right", "foot", "tongue"]` |
| `chn_pos` | np.ndarray \| None | ⚠️ | 通道位置 (n_channels, 3) | `None` 或 `np.array([[x,y,z], ...])` |
| `chn_ori` | np.ndarray \| None | ⚠️ | 通道方向 (n_channels, 3) | `None` 或 `np.array([[x,y,z], ...])` |
| `chn_type` | str | ✅ | 通道类型 | `"EEG"`, `"EOG"` |
| `montage` | str | ✅ | 电极配置 | `"10_20"`, `"custom"` |

**注意**:
- ⚠️ 标记的字段为可选，但**强烈建议**在分类任务中设置 `num_labels` 和 `category_list`
- `category_list` 的索引对应标签值（`category_list[0]` = 标签 0）

---

### 2. Trial Level (试验组)

**位置**: HDF5 文件中的组 `trial{trial_id}`

**用途**: 组织同一试验中的所有片段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `trial_id` | int | 试验唯一标识符 | `0`, `1`, `2`, ... |
| `session_id` | int | 会话标识符 | `1`, `2` (训练/测试) |

**命名规则**: `trial{trial_id}` (例如: `trial0`, `trial1`)

---

### 3. Segment Level (片段组和数据)

**位置**: `trial{trial_id}/segment{segment_id}/eeg`

**用途**: 存储实际的 EEG 数据片段及其元数据

#### Segment Group Attributes

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `segment_id` | int | 片段唯一标识符 | `0`, `1`, `2`, ... |
| `start_time` | float | 片段开始时间（秒） | `0.0`, `4.0`, `8.0` |
| `end_time` | float | 片段结束时间（秒） | `4.0`, `8.0`, `12.0` |
| `time_length` | float | 片段时长（秒） | `4.0` |
| `label` | np.ndarray | 标签数组（形状可变） | `np.array([0])`, `np.array([1, 0, 0])` |

#### EEG Dataset

- **数据类型**: `float32`
- **形状**: `[n_channels, n_timepoints]`
- **单位**: **微伏 (µV)**
- **命名**: `eeg`

**命名规则**: `segment{segment_id}` (例如: `segment0`, `segment1`)

---

## 数据单位规范

### 输入处理流程

```
原始文件 (V/mV/µV)
    ↓ [自动检测单位]
统一转换为伏特 (V) → MNE 处理
    ↓ [预处理: 滤波、重采样等]
转换为微伏 (µV) → 写入 HDF5
```

### 单位转换规则

| 原始单位 | 检测阈值 | 转换系数 | MNE 处理单位 |
|---------|---------|---------|------------|
| µV | > 0.01 | ÷ 1e6 | V |
| mV | 0.00001 ~ 0.01 | ÷ 1e3 | V |
| V | < 0.00001 | × 1 | V |

**输出单位**: 所有 HDF5 文件中的数据统一为 **µV (微伏)**

---

## 完整示例

### Python 代码示例

```python
from hdf5_io import HDF5Writer, HDF5Reader
from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
import numpy as np

# ===== 写入数据 =====
subject_attrs = SubjectAttrs(
    subject_id=1,
    dataset_name="BCIC2A_4class",
    task_type="motor_imaginary",
    downstream_task_type="classification",
    rsFreq=250.0,
    chn_name=["FZ", "FC3", "C3", "CZ", "C4", "FC4"],
    num_labels=4,
    category_list=["left", "right", "foot", "tongue"],
    chn_pos=None,
    chn_ori=None,
    chn_type="EEG",
    montage="10_20",
)

with HDF5Writer("sub_1.h5", subject_attrs) as writer:
    # 添加 trial
    trial_attrs = TrialAttrs(trial_id=0, session_id=1, task_name="motor_imagery")
    trial_name = writer.add_trial(trial_attrs)
    
    # 添加 segment
    eeg_data = np.random.randn(6, 1000).astype(np.float32)  # 6 channels, 1000 samples
    segment_attrs = SegmentAttrs(
        segment_id=0,
        start_time=0.0,
        end_time=4.0,
        time_length=4.0,
        label=np.array([0]),  # Label: "left"
        task_label="left_hand", # or None
    )
    writer.add_segment(trial_name, segment_attrs, eeg_data)

# ===== 读取数据 =====
with HDF5Reader("sub_1.h5") as reader:
    # 读取 subject 属性
    subj = reader.subject_attrs
    print(f"Dataset: {subj.dataset_name}")
    print(f"Labels: {subj.num_labels} classes")
    print(f"Categories: {subj.category_list}")
    
    # 遍历所有 segments
    for segment in reader.iter_segments():
        print(f"Trial {segment.trial.trial_id}, "
              f"Segment {segment.segment.segment_id}, "
              f"Label: {segment.segment.label}, "
              f"Shape: {segment.data.shape}")
```

---

## 设计原则

### ✅ 已实现的设计决策

1. **Supervised Only**: 仅支持监督学习数据
2. **命名规范**: 数据集名称 = 数据集名 + 实验类型 (例如: `"SEED_3class"`)
3. **术语统一**: 使用 "segment" 而非 "sample"
4. **时间信息**: Segment 包含 `start_time` 和 `end_time`
5. **下游任务**: 支持 `downstream_task_type` 字段
6. **标签元数据**: 支持 `num_labels` 和 `category_list`

### 📋 待实现功能

- [ ] 数据集分割策略 (`SplitStrategy`)
- [ ] 数据加载器中的分割函数
- [ ] 无监督数据支持（如需要）

---

## 字段对比表

| 文档版本 | 实际实现 | 状态 |
|---------|---------|------|
| `SampleAttrs` | `SegmentAttrs` | ✅ 已重命名 |
| `start_time`, `end_time` | ✅ 已实现 | ✅ 已添加 |
| `downstream_task_type` | ✅ 已实现 | ✅ 已添加 |
| `num_labels` | ✅ 已实现 | ⚠️ 文档未提及 |
| `category_list` | ✅ 已实现 | ⚠️ 文档未提及 |
| 数据单位规范 | ✅ 已实现 | ⚠️ 文档未提及 |

---

## 注意事项

1. **数据单位**: 所有 HDF5 文件中的数据统一为 **µV (微伏)**
2. **标签索引**: `category_list` 的索引对应标签值
   - `category_list[0]` = 标签 `0`
   - `category_list[1]` = 标签 `1`
   - ...
3. **时间单位**: `start_time` 和 `end_time` 单位为**秒**
4. **数据形状**: EEG 数据形状为 `[n_channels, n_timepoints]`
5. **必需字段**: 分类任务应设置 `num_labels` 和 `category_list`

---

## 相关文件

- `schema.py`: 数据结构定义
- `hdf5_io.py`: HDF5 读写实现
- `loader.py`: 数据加载器
- `config.py`: 配置和枚举类型

