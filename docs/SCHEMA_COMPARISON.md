# 数据结构文档对比分析

## 原始文档 vs 实际实现

### ✅ 已正确描述的部分

| 文档描述 | 实际实现 | 状态 |
|---------|---------|------|
| `SampleAttrs` → `SegmentAttrs` | ✅ 已实现为 `SegmentAttrs` | ✅ 正确 |
| 添加 `start_time`, `end_time` | ✅ 已实现 | ✅ 正确 |
| 添加 `downstream_task_type` | ✅ 已实现 | ✅ 正确 |
| 使用 "segment" 术语 | ✅ 已实现 | ✅ 正确 |
| 命名: `trial{trial_id}`, `segment{segment_id}` | ✅ 已实现 | ✅ 正确 |
| HDF5 层次结构 | ✅ 已实现 | ✅ 正确 |

### ⚠️ 文档缺失的重要信息

| 实际实现 | 文档状态 | 重要性 |
|---------|---------|--------|
| `num_labels` 字段 | ❌ 未提及 | ⭐⭐⭐ 高 |
| `category_list` 字段 | ❌ 未提及 | ⭐⭐⭐ 高 |
| 数据单位规范 (V/µV/mV) | ❌ 未提及 | ⭐⭐⭐ 高 |
| 单位自动检测逻辑 | ❌ 未提及 | ⭐⭐ 中 |
| 数据验证阈值 | ❌ 未提及 | ⭐⭐ 中 |

### 📋 待实现的功能（文档中提及）

| 文档描述 | 实际实现 | 状态 |
|---------|---------|------|
| Split strategies (`BY_SUBJECT`, `BY_SESSION`, etc.) | ⚠️ 部分实现 | 🔄 进行中 |
| `split_dataset()` 函数 | ⚠️ 部分实现 | 🔄 进行中 |
| `split_ratio` 参数 | ⚠️ 部分实现 | 🔄 进行中 |

---

## 改进建议

### 1. 更新原始文档

原始文档应补充以下内容：

```markdown
## HDF5 Structure (per subject: `sub_{sub_id}.h5`)
```
sub_0.h5
├── attrs (subject-level):
│     subject_id: int
│     dataset_name: str
│     task_type: str
│     downstream_task_type: str
│     rsFreq: float
│     chn_name, chn_pos, chn_ori, chn_type, montage
│     num_labels: int = 0              # NEW: 类别数量
│     category_list: list[str] = []   # NEW: 类别名称列表
│
├── /trial0 (group)
│   ├── attrs: trial_id, session_id
│   ├── /segment0 (group)
│   │   └── eeg (dataset)
│   │       attrs:
│   │         segment_id: int
│   │         start_time: float
│   │         end_time: float
│   │         time_length: float
│   │         label: array
│   │       data: float32[n_channels, n_timepoints]  # 单位: µV
│   └── /segmentN ...
└── /trialM ...
```

### 2. 添加数据单位规范

```markdown
## Data Unit Specification

- **Input**: Auto-detect unit (V/mV/µV) and convert to V for MNE processing
- **Processing**: MNE uses Volts (V) internally
- **Output**: All HDF5 files store data in **microvolts (µV)**
- **Conversion**: Automatic unit detection using robust statistics (percentile + MAD)
```

### 3. 添加标签元数据说明

```markdown
## Label Metadata

For classification tasks, set:
- `num_labels`: Number of classes (e.g., 4)
- `category_list`: List of class names, where `category_list[i]` corresponds to label `i`

Example:
```python
num_labels = 4
category_list = ["left", "right", "foot", "tongue"]
# label = 0 → "left"
# label = 1 → "right"
# label = 2 → "foot"
# label = 3 → "tongue"
```

---

## 文档完整性评分

| 类别 | 得分 | 说明 |
|------|------|------|
| 基本结构描述 | ✅ 100% | 层次结构、命名规范完整 |
| 字段完整性 | ⚠️ 70% | 缺少 `num_labels` 和 `category_list` |
| 数据单位规范 | ❌ 0% | 完全缺失 |
| 代码示例 | ⚠️ 50% | 有基本示例，但缺少关键字段 |
| 设计原则 | ✅ 100% | 设计决策描述清晰 |
| **总体评分** | **⚠️ 70%** | 需要补充重要信息 |

---

## 推荐行动

1. ✅ **立即更新**: 补充 `num_labels` 和 `category_list` 字段说明
2. ✅ **立即更新**: 添加数据单位规范章节
3. ✅ **立即更新**: 添加单位检测逻辑说明
4. 🔄 **后续完善**: 完成 split strategies 实现
5. 📝 **文档维护**: 保持文档与代码同步

---

## 参考文档

- `docs/HDF5_SCHEMA.md`: 完整的数据结构规范
- `docs/HDF5_STRUCTURE_DIAGRAM.md`: 可视化图表和示例
- `schema.py`: 实际的数据结构定义
- `hdf5_io.py`: HDF5 读写实现

