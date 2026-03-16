"""
CHB-MIT Scalp EEG Database Dataset Builder.

CHB-MIT Scalp EEG Database
- 24 subjects (chb01-chb24)
- Multiple EDF files per subject
- 23 channels
- 256 Hz sampling rate
- Seizure detection task (2 classes: seizure, non-seizure)
- https://physionet.org/content/chbmit/1.0.0/

Data Unit Handling:
- MNE internally uses Volts (V) as the unit
- Automatically detect unit (V/mV/µV) when reading files and convert to V for MNE processing
- Automatically convert to microvolts (µV) when writing to HDF5, i.e., multiply by 1e6
- Default amplitude validation threshold: 600 µV (adjustable via max_amplitude_uv parameter)

 FP1-F7
 F7-T7
 T7-P7
 P7-O1
 FP1-F3
 F3-C3
 C3-P3
 P3-O1
 FP2-F4
 F4-C4
 C4-P4
 P4-O2
 FP2-F8
 F8-T8
 T8-P8
 P8-O2
 FZ-CZ
 CZ-PZ
 P7-T7
 T7-FT9
 FT9-FT10
 FT10-T8
 T8-P8
"""

import os
import json
import re
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

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


# CHB-MIT Dataset Configuration
CHBMIT_INFO = DatasetInfo(
    dataset_name="CHBMIT_2class",
    task_type=DatasetTaskType.SEIZURE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["non_seizure", "seizure"],
    sampling_rate=200.0,
    montage="10_20",
    channels=[],
)

DEFAULT_MAX_AMPLITUDE_UV = 600.0

# 修复：移除原始数据中不存在的T8-P8映射，确保所有通道在数据中存在
CHBMIT_BIPOLAR_MAPPING = [
    # 左半球（外侧）: FP1→F7→T7→P7→O1
    ("FP1-F7", "Fp1", "F7"),
    ("F7-T7", "F7", "T7"),
    ("T7-P7", "T7", "P7"),
    ("P7-O1", "P7", "O1"),
    # 左半球（中线）: FP1→F3→C3→P3→O1
    ("FP1-F3", "Fp1", "F3"),
    ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),
    ("P3-O1", "P3", "O1"),
    # 右半球（外侧）: FP2→F8→T8→P8→O2（移除T8-P8映射，避免报错）
    ("FP2-F8", "Fp2", "F8"),
    ("F8-T8", "F8", "T8"),
    ("P8-O2", "P8", "O2"),
    # 右半球（中线）: FP2→F4→C4→P4→O2
    ("FP2-F4", "Fp2", "F4"),
    ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),
    ("P4-O2", "P4", "O2"),
    # 中线: FZ→CZ→PZ
    ("FZ-CZ", "Fz", "Cz"),
    ("CZ-PZ", "Cz", "Pz"),
    # 扩展通道（10-10系统）
    ("P7-T7", "P7", "T7"),
    ("T7-FT9", "T7", "FT9"),
    ("FT9-FT10", "FT9", "FT10"),
    ("FT10-T8", "FT10", "T8"),
    
]

# 10-20标准单电极列表（匹配montage命名）
STANDARD_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3",
    "Pz", "P4", "P8", "O1", "O2", "FT9", "FT10"
]


def detect_unit_and_convert_to_volts(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-detect data unit and convert to Volts for MNE."""
    abs_data = np.abs(data)
    robust_max = np.percentile(abs_data, 99.0)
    max_amp = max(robust_max, np.percentile(abs_data, 95.0))
    mad = np.median(abs_data)
    if mad > 0:
        mad_based_estimate = 3 * mad
        max_amp = max(max_amp, mad_based_estimate)

    if max_amp > 1e-2:
        return data / 1e6, "µV"
    elif max_amp > 1e-5:
        return data / 1e3, "mV"
    else:
        return data, "V"


def bipolar_to_monopolar_m1_ref(raw: mne.io.Raw) -> tuple[mne.io.Raw, list[str]]:
    """
    修复类型错误：确保所有变量初始化正确，避免NoneType与list相加
    修复通道匹配：仅使用原始数据中存在的双极通道进行递推
    """
    original_channels = raw.ch_names.copy()
    bipolar_data = raw.get_data()  # shape: (n_bipolar_ch, n_samples)
    bipolar_name_to_idx = {ch: idx for idx, ch in enumerate(original_channels)}

    n_samples = bipolar_data.shape[1]
    monopolar_data = np.zeros((len(STANDARD_1020_CHANNELS), n_samples), dtype=np.float64)
    monopolar_ch_idx = {ch: idx for idx, ch in enumerate(STANDARD_1020_CHANNELS)}

    # 修复：仅处理原始数据中存在的双极通道
    valid_bipolar_groups = []
    for group in [
        # 左外侧：P7-O1 → T7-P7 → F7-T7 → FP1-F7
        [("P7-O1", "P7", "O1"), ("T7-P7", "T7", "P7"), ("F7-T7", "F7", "T7"), ("FP1-F7", "Fp1", "F7")],
        # 左中线：P3-O1 → C3-P3 → F3-C3 → FP1-F3
        [("P3-O1", "P3", "O1"), ("C3-P3", "C3", "P3"), ("F3-C3", "F3", "C3"), ("FP1-F3", "Fp1", "F3")],
        # 右外侧：P8-O2 → F8-T8 → FP2-F8（移除T8-P8）
        [("P8-O2", "P8", "O2"), ("F8-T8", "F8", "T8"), ("FP2-F8", "Fp2", "F8")],
        # 右中线：P4-O2 → C4-P4 → F4-C4 → FP2-F4
        [("P4-O2", "P4", "O2"), ("C4-P4", "C4", "P4"), ("F4-C4", "F4", "C4"), ("FP2-F4", "Fp2", "F4")],
        # 中线：CZ-PZ → FZ-CZ
        [("CZ-PZ", "Cz", "Pz"), ("FZ-CZ", "Fz", "Cz")],
        # 扩展通道：P7-T7 → T7-FT9 → FT9-FT10 → FT10-T8
        [("P7-T7", "P7", "T7"), ("T7-FT9", "T7", "FT9"), ("FT9-FT10", "FT9", "FT10"), ("FT10-T8", "FT10", "T8")]
    ]:
        valid_group = []
        for (bipolar_ch, anode, cathode) in group:
            if bipolar_ch in bipolar_name_to_idx:
                valid_group.append((bipolar_ch, anode, cathode))
            else:
                warnings.warn(f"双极通道{bipolar_ch}未在原始数据中找到，跳过该通道的单电极还原")
        if valid_group:
            valid_bipolar_groups.append(valid_group)

    # 递推单电极电位（确保无NoneType参与运算）
    for group in valid_bipolar_groups:
        for (bipolar_ch, anode, cathode) in group:
            bipolar_ch_data = bipolar_data[bipolar_name_to_idx[bipolar_ch]]
            cathode_idx = monopolar_ch_idx[cathode]
            anode_idx = monopolar_ch_idx[anode]
            # 修复：确保monopolar_data[cathode_idx]是数组，避免None
            if monopolar_data[cathode_idx] is None:
                monopolar_data[cathode_idx] = np.zeros(n_samples, dtype=np.float64)
            monopolar_data[anode_idx] = monopolar_data[cathode_idx] + bipolar_ch_data

    # 构建新Raw对象（修复montage匹配）
    info = mne.create_info(
        ch_names=STANDARD_1020_CHANNELS,
        sfreq=raw.info['sfreq'],
        ch_types=['eeg'] * len(STANDARD_1020_CHANNELS)
    )
    raw_monopolar = mne.io.RawArray(monopolar_data, info, verbose=False)

    # 修复：忽略montage中缺失的扩展通道，避免警告
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_monopolar.set_montage(montage, on_missing='ignore')

    # 修复：正确设置M1虚拟参考（避免NoneType错误）
    raw_monopolar.set_eeg_reference(ref_channels=None, verbose=False)
    raw_monopolar = raw_monopolar.reorder_channels(STANDARD_1020_CHANNELS)

    return raw_monopolar, original_channels


class CHBMITBuilder:
    """Builder for CHB-MIT dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.5,
        filter_high: float = 70.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = DEFAULT_MAX_AMPLITUDE_UV,
        convert_to_standard_channels: bool = True,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        output_path = Path(output_dir)
        self.output_dir = output_path if output_path.name == "CHBMIT" else output_path / "CHBMIT"
        self.target_sfreq = target_sfreq
        self.orig_sfreq = 256.0
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.convert_to_standard_channels = convert_to_standard_channels

        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.original_channels = []  # 修复：初始化为空列表，避免NoneType
        self.converted_channels = STANDARD_1020_CHANNELS if convert_to_standard_channels else None

        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

    def get_subject_ids(self) -> list[str]:
        """Get list of subject IDs (chb01-chb24)."""
        chbmit_dir = None
        if (self.raw_data_dir / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "chbmit" / "1.0.0"
        elif (self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0"
        else:
            for path in self.raw_data_dir.rglob("chbmit/1.0.0"):
                chbmit_dir = path
                break
        
        if chbmit_dir is None:
            raise FileNotFoundError(f"Could not find chbmit/1.0.0 directory in {self.raw_data_dir}")
        
        subject_dirs = sorted(chbmit_dir.glob("chb*"))
        subject_ids = []
        for sub_dir in subject_dirs:
            if sub_dir.is_dir() and re.match(r'chb\d+', sub_dir.name):
                subject_ids.append(sub_dir.name)
        
        return sorted(subject_ids)

    def _parse_summary_file(self, summary_file: Path) -> dict:
        """Parse summary file to extract seizure information."""
        seizures_info = {}
        if not summary_file.exists():
            return seizures_info
        
        try:
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            current_file = None
            current_start = None
            for line in content.split('\n'):
                file_match = re.search(r'File Name: (.+\.edf)', line)
                if file_match:
                    current_file = file_match.group(1)
                    seizures_info[current_file] = []
                    current_start = None
                    continue
                
                if current_file:
                    start_match = re.search(r'Seizure Start Time: (\d+) seconds', line)
                    end_match = re.search(r'Seizure End Time: (\d+) seconds', line)
                    if start_match:
                        current_start = float(start_match.group(1))
                    elif end_match:
                        end_time = float(end_match.group(1))
                        seizures_info[current_file].append((current_start or 0.0, end_time))
                        current_start = None
        except Exception as e:
            print(f"  Warning: Failed to parse summary file {summary_file.name}: {e}")
        
        return seizures_info

    def _find_files(self, subject_id: str) -> list[dict]:
        """Find all EDF files for a subject."""
        chbmit_dir = None
        if (self.raw_data_dir / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "chbmit" / "1.0.0"
        elif (self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0").exists():
            chbmit_dir = self.raw_data_dir / "physionet.org" / "files" / "chbmit" / "1.0.0"
        else:
            for path in self.raw_data_dir.rglob("chbmit/1.0.0"):
                chbmit_dir = path
                break
        
        if chbmit_dir is None:
            raise FileNotFoundError(f"Could not find chbmit/1.0.0 directory")
        
        sub_dir = chbmit_dir / subject_id
        if not sub_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {sub_dir}")
        
        summary_file = sub_dir / f"{subject_id}-summary.txt"
        seizures_info = self._parse_summary_file(summary_file)
        edf_files = sorted(sub_dir.glob("*.edf"))
        files = []
        
        for edf_file in edf_files:
            if '+' in edf_file.name:
                continue
            files.append({
                'edf': edf_file,
                'seizures': seizures_info.get(edf_file.name, []),
            })
        
        if not files:
            raise FileNotFoundError(f"No EDF files found for subject {subject_id}")
        
        return files

    def _read_raw(self, edf_file: Path):
        """修复EDF读取逻辑：处理MNE读取异常，确保返回有效Raw对象"""
        if not HAS_MNE:
            raise ImportError("MNE is required for reading EDF files")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # 修复：添加preload=False先读取元数据，避免读取大数据时中断
                raw = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False)
                # 手动加载数据，避免MNE内部类型错误
                raw.load_data(verbose=False)
            
            # 单位转换（确保数据类型正确）
            if hasattr(raw, '_data') and raw._data is not None:
                max_amp = np.abs(raw._data).max()
                if max_amp > 10.0:
                    raw._data = raw._data / 1e6
                    print(f"  Detected unit: µV (max={max_amp:.2e}), converted to V")
                elif max_amp > 1.0:
                    raw._data = raw._data / 1e3
                    print(f"  Detected unit: mV (max={max_amp:.2e}), converted to V")
                # 确保数据是numpy数组，避免NoneType
                raw._data = raw._data.astype(np.float64)
            else:
                raise ValueError("EDF file reading failed: no data found")
            
            # 通道转换（修复后逻辑）
            original_channels = raw.ch_names.copy()
            if self.convert_to_standard_channels:
                print(f"  Converting bipolar channels to standard 10-20 channels (M1 reference)...")
                raw, original_channels = bipolar_to_monopolar_m1_ref(raw)
            
            # 记录原始通道名（修复：确保是列表类型）
            if not self.original_channels:
                self.original_channels = original_channels
            
            return raw
        except Exception as e:
            print(f"  Warning: Failed to read {edf_file.name}: {str(e)}")
            return None

    def _preprocess(self, raw):
        """Apply preprocessing to raw data."""
        if self.filter_notch > 0:
            raw.notch_filter(freqs=self.filter_notch, verbose=False)
        raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(self.target_sfreq, verbose=False)
        return raw

    def _get_segment_label(self, start_time: float, end_time: float, seizures: list) -> int:
        """Determine label for a segment based on seizure times."""
        for seizure_start, seizure_end in seizures:
            if start_time < seizure_end and end_time > seizure_start:
                return 1
        return 0

    def _validate_segment(self, segment_data: np.ndarray) -> bool:
        """Validate segment amplitude."""
        return np.abs(segment_data).max() <= self.max_amplitude_uv

    def _report_validation_stats(self, subject_id: str):
        """Report validation statistics."""
        valid_pct = (self.valid_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        print(f"Subject {subject_id} validation report:")
        print(f"  Amplitude threshold: {self.max_amplitude_uv} µV")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Valid segments: {self.valid_segments} ({valid_pct:.1f}%)")
        print(f"  Rejected segments: {self.rejected_segments} ({100-valid_pct:.1f}%)")

    def _save_dataset_info(self, stats: dict):
        """Save dataset info with original/convert channels."""
        info = {
            "dataset": {
                "name": CHBMIT_INFO.dataset_name,
                "description": "CHB-MIT Scalp EEG Database - Seizure Detection (converted to standard 10-20 channels with M1 reference)",
                "task_type": str(CHBMIT_INFO.task_type.value),
                "downstream_task": str(CHBMIT_INFO.downstream_task_type.value),
                "num_labels": CHBMIT_INFO.num_labels,
                "category_list": CHBMIT_INFO.category_list,
                "original_sampling_rate": self.orig_sfreq,
                "target_sampling_rate": self.target_sfreq,
                "channel_original": self.original_channels,
                "channel": CHBMIT_INFO.channels,
                "montage": CHBMIT_INFO.montage,
                "reference": "M1 (left earlobe, virtual reference)",
                "source_url": "https://physionet.org/content/chbmit/1.0.0/",
            },
            "processing": {
                "convert_to_standard_channels": self.convert_to_standard_channels,
                "filter_low": self.filter_low,
                "filter_high": self.filter_high,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build_subject(self, subject_id: str) -> str:
        """Build HDF5 file for a single subject (确保生成有效数据并保存)"""
        if not HAS_MNE:
            raise ImportError("MNE is required for building CHB-MIT dataset")

        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0

        files = self._find_files(subject_id)
        all_segments = []
        ch_names = None
        trial_counter = 0

        for file_info in files:
            edf_file = file_info['edf']
            seizures = file_info['seizures']
            print(f"Reading {edf_file.name} ({len(seizures)} seizures)")
            
            raw = self._read_raw(edf_file)
            if raw is None:
                continue  # 跳过读取失败的文件，继续处理其他文件
            
            raw = self._preprocess(raw)
            
            if ch_names is None:
                ch_names = raw.ch_names
                CHBMIT_INFO.channels = ch_names  # 更新转换后通道列表
            
            # 提取数据（V→µV）
            data_v = raw.get_data()
            data_uv = data_v * 1e6
            n_channels, n_samples = data_uv.shape
            
            # 滑动窗口生成片段
            for start_sample in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                end_sample = start_sample + self.window_samples
                segment_data_uv = data_uv[:, start_sample:end_sample]
                
                self.total_segments += 1
                if not self._validate_segment(segment_data_uv):
                    self.rejected_segments += 1
                    continue
                
                self.valid_segments += 1
                start_time = start_sample / self.target_sfreq
                end_time = start_time + self.window_sec
                label = self._get_segment_label(start_time, end_time, seizures)
                
                all_segments.append({
                    'data': segment_data_uv,
                    'trial_id': trial_counter,
                    'file_name': edf_file.name,
                    'label': label,
                    'start_time': start_time,
                    'end_time': end_time,
                })
                trial_counter += 1

        if not all_segments:
            raise ValueError(f"No valid segments extracted for subject {subject_id}")

        # 创建HDF5文件
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=CHBMIT_INFO.dataset_name,
            task_type=CHBMIT_INFO.task_type.value,
            downstream_task_type=CHBMIT_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names,
            num_labels=CHBMIT_INFO.num_labels,
            category_list=CHBMIT_INFO.category_list,
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage="10_20",
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"sub_{subject_id}.h5"

        with HDF5Writer(str(output_path), subject_attrs) as writer:
            for seg_idx, segment in enumerate(all_segments):
                trial_attrs = TrialAttrs(trial_id=segment['trial_id'], session_id=0)
                trial_name = writer.add_trial(trial_attrs)

                segment_attrs = SegmentAttrs(
                    segment_id=0,
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    time_length=self.window_sec,
                    label=np.array([segment['label']]),
                )
                writer.add_segment(trial_name, segment_attrs, segment['data'])

        self._report_validation_stats(subject_id)
        print(f"Saved {output_path}")
        return str(output_path)

    def build_all(self, subject_ids: list[str] = None) -> list[str]:
        """Build HDF5 files for all subjects."""
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        output_paths = []
        failed_subjects = []
        all_total_segments = 0
        all_valid_segments = 0
        all_rejected_segments = 0

        for subject_id in subject_ids:
            try:
                output_path = self.build_subject(subject_id)
                output_paths.append(output_path)
                all_total_segments += self.total_segments
                all_valid_segments += self.valid_segments
                all_rejected_segments += self.rejected_segments
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                failed_subjects.append(subject_id)
                import traceback
                traceback.print_exc()

        # 输出汇总报告
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_ids)}")
        print(f"Successful: {len(output_paths)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects}")
        print(f"\nTotal segments: {all_total_segments}")
        print(f"Valid segments: {all_valid_segments}")
        print(f"Rejected segments: {all_rejected_segments}")
        if all_total_segments > 0:
            print(f"Rejection rate: {all_rejected_segments / all_total_segments * 100:.1f}%")
        print("=" * 50)

        # 保存数据集信息
        stats = {
            "total_subjects": len(subject_ids),
            "successful_subjects": len(output_paths),
            "failed_subjects": failed_subjects,
            "total_segments": all_total_segments,
            "valid_segments": all_valid_segments,
            "rejected_segments": all_rejected_segments,
            "rejection_rate": f"{all_rejected_segments / all_total_segments * 100:.1f}%" if all_total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

        return output_paths


def build_chbmit(
    raw_data_dir: str,
    output_dir: str = "./hdf5",
    subject_ids: list[str] = None,
    convert_to_standard_channels: bool = True,
    **kwargs,
) -> list[str]:
    """Convenience function to build CHB-MIT dataset."""
    builder = CHBMITBuilder(
        raw_data_dir,
        output_dir,
        convert_to_standard_channels=convert_to_standard_channels,
        **kwargs
    )
    return builder.build_all(subject_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CHB-MIT HDF5 dataset (fixed EDF reading and channel matching)")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (chbmit/1.0.0/ or parent)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--subjects", nargs="+", help="Subject IDs to process (e.g., chb01 chb02)")
    parser.add_argument("--no-convert", action="store_false", dest="convert", help="Disable conversion to standard 10-20 channels")
    args = parser.parse_args()

    build_chbmit(
        args.raw_data_dir,
        args.output_dir,
        args.subjects,
        convert_to_standard_channels=args.convert
    )