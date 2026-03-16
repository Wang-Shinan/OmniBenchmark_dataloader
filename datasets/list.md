 由于您提供的 DOI 链接 `arXiv:2508.17742` 属于 2025 年 8 月发表的论文（可能与之前论文属于同一系列或同类研究），该文对 EEG 基础模型的基准测试进行了更深度的扩展。

结合之前 PDF 文件中的数据集（23个）与新论文中涉及的数据，取并集后涵盖了目前 EEG 研究领域**最完整、最主流的公开数据集清单**。

---

### **EEG 全球主流公开数据集下载并集清单**

我们将这些数据集按功能 paradigm 分类，并提供最直接的下载入口：

#### **1. 运动相关（运动想象 MI / 运动执行 ME）**

这是基础模型评估泛化能力最常用的领域。
| 数据集名称 | 特点 | 下载链接/方式 |
| --- | --- | --- |
| **EEG-MI-BCI (Cho2017)** | 52人, 2类 MI, 64通道 | [GigaDB 100295](http://gigadb.org/dataset/100295) |
| **BCI-IV-2a** | 9人, 4类 MI, 经典基准 | [BNCI Horizon](http://bnci-horizon-2020.eu/database/data-sets) |
| **BCI-IV-2b** | 9人, 2类 MI, 3通道 | [BNCI Horizon](http://bnci-horizon-2020.eu/database/data-sets) |
| **HGD (High Gamma)** | 20人, 4类 ME, 高采样率 | [MNE-Datasets](https://www.google.com/search?q=https://mne.tools/stable/generated/mne.datasets.fetch_hgd.html) |
| **Large-MI-Classic** | 13人, 6类动作, 规模大 | [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/) |
| **Large-MI-5F** | 13人, 5类手指运动想象 | [PhysioNet 获取](https://physionet.org/content/eegmmidb/) |
| **Online MI BCI (Stieger)** | 62人, 600小时长记录 | [Scientific Data (Stieger 2021)](https://doi.org/10.1038/s41597-021-00883-1) |
| **Upper-limb ME/MI** | 15人, 7类上肢复杂动作 | [Ofner 2017 数据存储](https://doi.org/10.1371/journal.pone.0182578) |

#### **2. 临床、睡眠与医疗诊断**

用于预训练学习“脑电生理共性”的核心数据池。
| 数据集名称 | 特点 | 下载链接/方式 |
| --- | --- | --- |
| **TUEG / TUEV** | **世界最大**, 1.5TB 临床数据 | [Temple University Hospital (注册申请)](https://isip.piconepress.com/projects/tuh_eeg/) |
| **Alzheimer's (AD/FTD)** | 痴呆症与健康对照分类 | [Mendeley Data (Miltiadous)](https://www.google.com/search?q=https://data.mendeley.com/datasets/7p6pxtdt2c/1) |
| **Bonn Seizure** | 经典癫痫分类基准 | [University of Bonn 官网](https://www.google.com/search?q=http://epileptologie-bonn.de/cms/front_content.php%3Fidcat%3D193) |
| **ISRUC-Sleep** | 睡眠分期标准数据集 | [Zenodo 或 ISRUC 官网](https://www.google.com/search?q=https://isruc.isr.uc.pt/) |
| **SHHS** | 500万个睡眠 EEG 片段 | [National Sleep Research Resource](https://sleepdata.org/datasets/shhs) |

#### **3. 情感识别与内部语言（Inner Speech）**

| 数据集名称 | 特点 | 下载链接/方式 |
| --- | --- | --- |
| **FACED** | 123人, 9类细粒度情绪 | [Science Data (Chen 2023a)](https://www.google.com/search?q=https://doi.org/10.1038/s41597-023-02640-5) |
| **SEED (V / VIG)** | 上海交大经典情感/警觉度 | [BCMI SJTU (签署协议申请)](https://bcmi.sjtu.edu.cn/home/seed/index.html) |
| **Thinking Out Loud** | 内部语言/默念单词识别 | [OpenNeuro ds003626](https://openneuro.org/datasets/ds003626) |

#### **4. 视觉 (SSVEP) 与 听觉 (Auditory)**

| 数据集名称 | 特点 | 下载链接/方式 |
| --- | --- | --- |
| **BETA** | 大规模 SSVEP 基准数据 | [Figshare (Liu 2020)](https://www.google.com/search?q=https://doi.org/10.6084/m9.figshare.10066937) |
| **Binocular SSVEP** | 40类双目诱发频率组合 | [GigaDB 102557](https://www.google.com/search?q=http://gigadb.org/dataset/102557) |
| **DTU Cocktail Party** | 听觉注意力连续解码 | [DTU Data (Fuglsang)](https://www.google.com/search?q=https://doi.org/10.11583/DTU.5699944.v1) |
| **KUL Auditory** | 鲁汶大学大规模听觉语料 | [Zenodo (Bollens 2023)](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.7428800) |

#### **5. 诱发电位 (ERP) 与 错误监控 (ERN)**

| 数据集名称 | 特点 | 下载链接/方式 |
| --- | --- | --- |
| **P300 Speller (Won)** | 100+人大型 P300 数据集 | [Scientific Data (Won 2022)](https://www.google.com/search?q=https://doi.org/10.1038/s41597-022-01502-8) |
| **Error-related (ERN)** | 机器人交互中的错误监控电位 | [Kueper 2024 数据集](https://doi.org/10.3389/fnhum.2024.1304311) |

---

### **核心工具推荐：如何快速搞定这些数据？**

如果您在做基础模型或深度学习研究，**千万不要一个一个去网页下载**，效率极低。

1. **首选：MOABB (Mother of All BCI Benchmarks)**
* **作用**：上述清单中约 80% 的数据集已集成在 MOABB 中。
* **方法**：`pip install moabb`。在 Python 中直接调用 `dataset = Cho2017()` 后运行 `dataset.get_data()`，系统会自动寻找最优镜像下载、解压并处理成标准的 `mne.Epochs` 格式。


2. **大厂官方 API**
* **PhysioNet**：使用命令行工具 `wget -r -np https://physionet.org/files/eegmmidb/1.0.0/` 可以一键抓取全部 109 名受试者的数据。
* **TUH EEG**：需在官网注册，之后使用其提供的专有 `rsync` 协议进行断点续传。



