# Metadata Module Design Plan

## 1. 目标
构建一个标准化的框架，用于提取、存储和管理 EEG 数据集的人口学信息（Demographics）和采集背景信息（Acquisition Context），并将其独立存储为 HDF5 格式，以便于后续的统计分析和偏差研究。

## 2. 目录结构
在 `benchmark_dataloader` 下新增 `metadata` 模块：

```
benchmark_dataloader/
├── metadata/
│   ├── __init__.py
│   ├── schema.py          # [Core] 定义元数据的数据结构 (DataClasses)
│   ├── io.py              # [Core] 负责元数据 HDF5 的读写操作
│   ├── utils.py           # [Helper] 通用工具 (e.g., age parsing, gender normalization)
│   └── extractors/        # [Impl] 针对特定数据集的提取脚本
│       ├── __init__.py
│       ├── base.py        # [Abstract] 定义提取器的基类接口
│       ├── seed.py        # [Example] SEED 数据集提取器
│       ├── adhd.py        # [Example] ADHD 数据集提取器
│       └── ...
```

## 3. Schema 设计 (`schema.py`)

我们将定义两个核心的数据类（DataClass）：

### 3.1 Demographics (人口学信息)
用于描述受试者的个人属性。
- `subject_id`: str (Standardized ID)
- `age`: float (Years)
- `gender`: str (Standardized: 'M', 'F', 'NB', 'Unknown')
- `handedness`: str ('R', 'L', 'Ambidextrous', 'Unknown')
- `group`: str (e.g., 'Healthy', 'Patient', 'Expert')
- `diagnosis`: list[str] (For clinical datasets, e.g., ['ADHD', 'Depression'])
- `education`: int (Years of education, optional)
- `notes`: str (Any additional remarks)

### 3.2 AcquisitionContext (采集背景)
用于描述数据采集时的环境和设备状态。
- `date`: str (ISO 8601 format, optional)
- `time`: str (HH:MM, optional)
- `location`: str (e.g., 'Lab-A', 'Hospital-Room-3')
- `device_model`: str (e.g., 'Neuroscan SynAmps2', 'Emotiv EPOC')
- `cap_layout`: str (e.g., '10-20', '10-10')
- `reference`: str (e.g., 'A1+A2', 'Cz')
- `ground`: str
- `sampling_rate`: float (Nominal Hz)
- `filter_settings`: dict (Hardware filters applied during recording)
- `impedance_check`: bool (Whether impedance was checked)
- `environment_notes`: str (Temperature, noise level, etc.)

## 4. HDF5 存储结构 (`io.py`)

每个数据集生成一个独立的 `metadata.h5` 文件（或 `{dataset_name}_meta.h5`）。
文件内部结构如下：

```
/ (Root)
├── dataset_info (Attributes)
│   ├── name: "SEED"
│   ├── version: "1.0"
│   └── last_updated: "2023-10-27"
│
└── subjects (Group)
    ├── sub_001 (Group)
    │   ├── demographics (Group/Attributes)
    │   │   ├── age: 24
    │   │   ├── gender: "F"
    │   │   └── ...
    │   └── context (Group/Attributes)
    │       ├── device: "Neuroscan"
    │       └── ...
    ├── sub_002
    │   ...
```

## 5. 工作流 (Workflow)

1.  **实现 Schema**: 在 `metadata/schema.py` 中定义上述 DataClass。
2.  **实现 Base Extractor**: 在 `metadata/extractors/base.py` 中定义基类，强制子类实现 `extract_subject_info(subject_id)` 方法。
3.  **实现具体 Extractor**: 针对每个数据集（如 SEED），编写逻辑从原始文件（如 `readme.txt`, `.mat` header, Excel 表格）中解析信息。
4.  **构建**: 运行构建脚本，调用 Extractor，使用 `MetadataWriter` 将对象写入 HDF5。

## 6. 下一步行动计划
1. 创建 `schema.py` 和 `io.py`。
2. 定义 `BaseExtractor`。
3. 为现有数据集（如 ADHD 或 SEED）编写一个示例提取器。
