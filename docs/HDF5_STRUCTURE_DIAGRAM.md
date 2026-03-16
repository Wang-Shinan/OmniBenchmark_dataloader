# HDF5 数据结构可视化图表

## 1. 层次结构树状图

```
sub_1.h5 (HDF5 File)
│
├─ 📋 Root Attributes (SubjectAttrs)
│  │
│  ├─ subject_id: 1
│  ├─ dataset_name: "BCIC2A_4class"
│  ├─ task_type: "motor_imaginary"
│  ├─ downstream_task_type: "classification"
│  ├─ rsFreq: 250.0
│  ├─ chn_name: ["FZ", "FC3", "C3", "CZ", "C4", "FC4"]
│  ├─ num_labels: 4
│  ├─ category_list: ["left", "right", "foot", "tongue"]
│  ├─ chn_pos: None
│  ├─ chn_ori: None
│  ├─ chn_type: "EEG"
│  └─ montage: "10_20"
│
├─ 📁 trial0 (Group)
│  │
│  ├─ 📋 Attributes (TrialAttrs)
│  │  ├─ trial_id: 0
│  │  └─ session_id: 1
│  │
│  ├─ 📁 segment0 (Group)
│  │  └─ 📊 eeg (Dataset: float32[6, 1000])
│  │     ├─ 📋 Attributes (SegmentAttrs)
│  │     │  ├─ segment_id: 0
│  │     │  ├─ start_time: 0.0
│  │     │  ├─ end_time: 4.0
│  │     │  ├─ time_length: 4.0
│  │     │  └─ label: [0]
│  │     └─ 📈 Data: EEG signal (µV)
│  │
│  ├─ 📁 segment1 (Group)
│  │  └─ 📊 eeg (Dataset: float32[6, 1000])
│  │     └─ ...
│  │
│  └─ 📁 segmentN (Group)
│     └─ ...
│
├─ 📁 trial1 (Group)
│  └─ ...
│
└─ 📁 trialM (Group)
   └─ ...
```

---

## 2. Mermaid 图表

```mermaid
graph TB
    subgraph HDF5["sub_1.h5 (HDF5 File)"]
        Root["📋 Root Attributes<br/>(SubjectAttrs)"]
        
        Root --> |"subject_id: 1"| Attr1["ID"]
        Root --> |"dataset_name"| Attr2["BCIC2A_4class"]
        Root --> |"task_type"| Attr3["motor_imaginary"]
        Root --> |"downstream_task_type"| Attr4["classification"]
        Root --> |"rsFreq"| Attr5["250.0 Hz"]
        Root --> |"num_labels"| Attr6["4"]
        Root --> |"category_list"| Attr7["left, right, foot, tongue"]
        
        Root --> Trial0["📁 trial0"]
        Root --> Trial1["📁 trial1"]
        Root --> TrialM["📁 trialM"]
        
        Trial0 --> |"trial_id: 0<br/>session_id: 1"| Seg0["📁 segment0"]
        Trial0 --> Seg1["📁 segment1"]
        Trial0 --> SegN["📁 segmentN"]
        
        Seg0 --> |"eeg dataset"| Data0["📊 EEG Data<br/>float32[6, 1000]<br/>单位: µV"]
        Seg0 --> |"segment_id: 0<br/>start_time: 0.0<br/>end_time: 4.0<br/>label: [0]"| Meta0["📋 Segment Metadata"]
    end
    
    style Root fill:#e1f5ff
    style Trial0 fill:#fff4e1
    style Seg0 fill:#e8f5e9
    style Data0 fill:#fce4ec
    style Meta0 fill:#f3e5f5
```

---

## 3. 数据流图

```mermaid
flowchart LR
    A["原始数据文件<br/>(.mat, .edf, .gdf)"] --> B["单位检测<br/>(V/mV/µV)"]
    B --> C["转换为伏特 (V)"]
    C --> D["MNE 预处理<br/>(滤波、重采样)"]
    D --> E["转换为微伏 (µV)"]
    E --> F["写入 HDF5<br/>sub_X.h5"]
    
    F --> G["SubjectAttrs<br/>(根属性)"]
    F --> H["Trial Groups<br/>(trial0, trial1, ...)"]
    H --> I["Segment Groups<br/>(segment0, segment1, ...)"]
    I --> J["EEG Dataset<br/>(数据 + 属性)"]
    
    style A fill:#ffebee
    style C fill:#e3f2fd
    style E fill:#e8f5e9
    style F fill:#fff3e0
    style J fill:#f3e5f5
```

---

## 4. 属性继承关系

```mermaid
graph TD
    Subject["SubjectAttrs<br/>├─ subject_id<br/>├─ dataset_name<br/>├─ task_type<br/>├─ num_labels<br/>└─ category_list"] 
    
    Trial["TrialAttrs<br/>├─ trial_id<br/>└─ session_id"]
    
    Segment["SegmentAttrs<br/>├─ segment_id<br/>├─ start_time<br/>├─ end_time<br/>├─ time_length<br/>└─ label"]
    
    EEGData["EEG Data<br/>float32[n_channels, n_timepoints]<br/>单位: µV"]
    
    Subject --> Trial
    Trial --> Segment
    Segment --> EEGData
    
    EEGSegment["EEGSegment<br/>(完整数据对象)"]
    Subject -.->|"继承"| EEGSegment
    Trial -.->|"继承"| EEGSegment
    Segment -.->|"继承"| EEGSegment
    EEGData -.->|"包含"| EEGSegment
    
    style Subject fill:#e1f5ff
    style Trial fill:#fff4e1
    style Segment fill:#e8f5e9
    style EEGData fill:#fce4ec
    style EEGSegment fill:#f3e5f5
```

---

## 5. 字段类型和约束表

### SubjectAttrs (根属性)

| 字段 | 类型 | 必需 | 默认值 | 约束 | 示例 |
|------|------|------|--------|------|------|
| `subject_id` | int \| str | ✅ | - | 唯一标识符 | `1`, `"sub_001"` |
| `dataset_name` | str | ✅ | - | 非空字符串 | `"BCIC2A_4class"` |
| `task_type` | str | ✅ | - | 枚举值 | `"motor_imaginary"` |
| `downstream_task_type` | str | ✅ | - | 枚举值 | `"classification"` |
| `rsFreq` | float | ✅ | - | > 0 | `250.0` |
| `chn_name` | list[str] | ✅ | - | 非空列表 | `["FZ", "FC3"]` |
| `num_labels` | int | ⚠️ | `0` | ≥ 0 | `4` |
| `category_list` | list[str] | ⚠️ | `[]` | len = num_labels | `["left", "right"]` |
| `chn_pos` | np.ndarray \| None | ⚠️ | `None` | shape: (n, 3) | `None` |
| `chn_ori` | np.ndarray \| None | ⚠️ | `None` | shape: (n, 3) | `None` |
| `chn_type` | str | ✅ | - | 非空字符串 | `"EEG"` |
| `montage` | str | ✅ | - | 非空字符串 | `"10_20"` |

### TrialAttrs (Trial 组属性)

| 字段 | 类型 | 必需 | 约束 | 示例 |
|------|------|------|------|------|
| `trial_id` | int | ✅ | ≥ 0, 唯一 | `0`, `1`, `2` |
| `session_id` | int | ✅ | ≥ 0 | `1`, `2` |

### SegmentAttrs (Segment 组属性)

| 字段 | 类型 | 必需 | 约束 | 示例 |
|------|------|------|------|------|
| `segment_id` | int | ✅ | ≥ 0, 在 trial 内唯一 | `0`, `1`, `2` |
| `start_time` | float | ✅ | ≥ 0 | `0.0`, `4.0` |
| `end_time` | float | ✅ | > start_time | `4.0`, `8.0` |
| `time_length` | float | ✅ | = end_time - start_time | `4.0` |
| `label` | np.ndarray | ✅ | 形状可变 | `[0]`, `[1, 0, 0]` |

### EEG Dataset

| 属性 | 类型 | 约束 | 示例 |
|------|------|------|------|
| 数据类型 | `float32` | - | - |
| 形状 | `[n_channels, n_timepoints]` | n_channels > 0, n_timepoints > 0 | `[6, 1000]` |
| 单位 | µV (微伏) | - | - |
| 命名 | `eeg` | - | - |

---

## 6. 访问路径示例

### Python 访问路径

```python
# 文件级别
file = h5py.File("sub_1.h5", "r")

# Subject 属性
subject_id = file.attrs["subject_id"]
num_labels = file.attrs["num_labels"]
category_list = file.attrs["category_list"]

# Trial 组
trial0 = file["trial0"]
trial_id = trial0.attrs["trial_id"]
session_id = trial0.attrs["session_id"]

# Segment 组
segment0 = trial0["segment0"]
segment_id = segment0["eeg"].attrs["segment_id"]
start_time = segment0["eeg"].attrs["start_time"]
end_time = segment0["eeg"].attrs["end_time"]
label = segment0["eeg"].attrs["label"]

# EEG 数据
eeg_data = segment0["eeg"][:]  # shape: [n_channels, n_timepoints]
```

### 路径字符串

```
sub_1.h5
├─ /attrs["subject_id"]
├─ /attrs["num_labels"]
├─ /attrs["category_list"]
├─ /trial0/attrs["trial_id"]
├─ /trial0/attrs["session_id"]
├─ /trial0/segment0/eeg (dataset)
│  ├─ /trial0/segment0/eeg/attrs["segment_id"]
│  ├─ /trial0/segment0/eeg/attrs["start_time"]
│  ├─ /trial0/segment0/eeg/attrs["end_time"]
│  ├─ /trial0/segment0/eeg/attrs["label"]
│  └─ /trial0/segment0/eeg[:] (data array)
└─ /trial0/segment1/eeg ...
```

---

## 7. 数据单位转换流程图

```mermaid
flowchart TD
    Start["原始数据文件"] --> Detect["检测单位<br/>(使用百分位数+MAD)"]
    
    Detect -->|"max_amp > 0.01"| UV["µV<br/>÷ 1e6"]
    Detect -->|"0.00001 < max_amp ≤ 0.01"| MV["mV<br/>÷ 1e3"]
    Detect -->|"max_amp ≤ 0.00001"| V["V<br/>× 1"]
    
    UV --> MNE["MNE 处理<br/>(单位: V)"]
    MV --> MNE
    V --> MNE
    
    MNE --> Filter["滤波<br/>(bandpass, notch)"]
    Filter --> Resample["重采样<br/>(target_sfreq)"]
    Resample --> Convert["转换为 µV<br/>× 1e6"]
    Convert --> Save["保存到 HDF5<br/>(单位: µV)"]
    
    style Detect fill:#e3f2fd
    style MNE fill:#fff3e0
    style Save fill:#e8f5e9
```

---

## 8. 标签映射示例

### BCIC-2A 数据集示例

```python
# SubjectAttrs 中定义
num_labels = 4
category_list = ["left", "right", "foot", "tongue"]

# Segment 中的标签
label = np.array([0])  # → "left"
label = np.array([1])  # → "right"
label = np.array([2])  # → "foot"
label = np.array([3])  # → "tongue"
```

### 标签索引映射

```
category_list[0] = "left"   ← label = 0
category_list[1] = "right"  ← label = 1
category_list[2] = "foot"   ← label = 2
category_list[3] = "tongue" ← label = 3
```

---

## 总结

- ✅ **层次结构**: Subject → Trial → Segment → EEG Data
- ✅ **元数据完整**: 每个层级都有相应的属性
- ✅ **单位统一**: 所有数据统一为 µV
- ✅ **标签支持**: 支持 `num_labels` 和 `category_list`
- ✅ **时间信息**: Segment 包含完整的时间信息
- ✅ **类型安全**: 使用 dataclass 确保类型一致性

