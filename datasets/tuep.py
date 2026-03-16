"""
TUEP (Temple University Hospital EEG Epilepsy) Dataset Builder.

Directory Structure (similar to TUAB):
root/
  train/
    seizure/01_tcp_ar/file.edf
    non_seizure/01_tcp_ar/file.edf
  eval/
    seizure/01_tcp_ar/file.edf
    non_seizure/01_tcp_ar/file.edf

Each file is named like: aaaaaaaq_s004_t000.edf
Subject ID: aaaaaaaq
"""

from pathlib import Path
import re
import json
import csv
import numpy as np
from collections import defaultdict
from datetime import datetime
import warnings

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from ..utils import ElectrodeSet
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import ElectrodeSet
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType

TUEP_INFO = DatasetInfo(
    dataset_name="TUEP",
    task_type=DatasetTaskType.SEIZURE,  # Epilepsy/Seizure Detection
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=["non_seizure", "seizure"],
    sampling_rate=200.0,
    montage="10_20",
    channels=ElectrodeSet.Standard_10_20,
)

# Canonical EEG channels based on TUH documentation (10-20 scalp electrodes).
# We drop A1/A2 here because它们主要作为参考电极出现。
CANONICAL_EEG_CHANNELS = [
    ch for ch in ElectrodeSet.Standard_10_20 if ch not in ("A1", "A2")
]

class TUEPBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5",
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_notch: float = 60.0,
        max_amplitude_uv: float = None, # Disable amplitude filtering
        clip_threshold: float = None,
        tcp_mode: str | None = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        # output_dir 由外部控制，是否带 tcp 前缀由调用方决定
        self.output_dir = Path(output_dir) / "TUEP"
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.filter_notch = filter_notch
        self.max_amplitude_uv = max_amplitude_uv
        self.clip_threshold = clip_threshold
        # If specified, only process files whose path contains this TCP montage
        # e.g., "01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"
        self.tcp_mode = tcp_mode
        
        self.window_samples = int(window_sec * target_sfreq)
        self.stride_samples = int(stride_sec * target_sfreq)
        self.electrode_set = ElectrodeSet()
        
        # Track validation statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        
        # Store actual channels from data (will be set during first subject processing)
        self._dataset_channels = list(CANONICAL_EEG_CHANNELS)

    def _find_files(self):
        """Find all EDF files and group by subject."""
        files = list(self.raw_data_dir.rglob("*.edf"))
        subject_map = defaultdict(list)
        
        for path in files:
            parts = path.parts

            # Optional TCP montage filtering based on directory name, e.g. "02_tcp_le"
            if self.tcp_mode is not None:
                if self.tcp_mode not in parts:
                    continue

            # Check path components for split and label
            # Path expected: .../split/label/01_tcp_ar/filename.edf
            # We look for 'train'/'eval' and 'seizure'/'non_seizure' in parts
            split = "unknown"
            if "train" in parts:
                split = "train"
            elif "eval" in parts:
                split = "eval"
                
            label_str = "unknown"
            # Simple directory-based labeling:
            # - 00_epilepsy -> seizure (label=1)
            # - 01_no_epilepsy -> non_seizure (label=0)
            parts_str = ' '.join(str(p).lower() for p in parts)
            if "00_epilepsy" in parts_str:
                label_str = "seizure"
            elif "01_no_epilepsy" in parts_str:
                label_str = "non_seizure"
            else:
                # Legacy fallback for other naming conventions
                if "seizure" in parts_str and "non" not in parts_str:
                    label_str = "seizure"
                elif "non_seizure" in parts_str or "non-seizure" in parts_str or "no_seizure" in parts_str or "no-seizure" in parts_str:
                    label_str = "non_seizure"
                elif "epilepsy" in parts_str and "no" not in parts_str and "non" not in parts_str:
                    label_str = "seizure"
                elif "epilepsy" in parts_str:
                    label_str = "non_seizure"
                
            if label_str == "unknown":
                print(f"Skipping {path.name}: unknown label (path: {path.parent})")
                continue

            # Parse filename for Subject ID
            # aaaaaaaq_s004_t000.edf -> aaaaaaaq
            stem = path.stem
            sub_id_match = re.match(r'([a-zA-Z0-9]+)_s\d+_t\d+', stem)
            if sub_id_match:
                sub_id = sub_id_match.group(1)
            else:
                # Fallback: take first part before _
                sub_id = stem.split('_')[0]
                
            subject_map[sub_id].append({
                "path": path,
                "split": split,
                "label": label_str,
                "stem": stem
            })
            
        return subject_map

    def _collect_common_channels(self, subject_map):
        """
        根据 TUH DOCS 直接指定 canonical EEG 通道集合。

        注意：
        - 不再根据数据做全局交集统计；
        - 直接使用 CANONICAL_EEG_CHANNELS（基于 10-20 + 去掉 A1/A2）；
        - 缺失的少量通道在 _process_file 中用 0 填补，而不是整条记录跳过。
        """
        self._dataset_channels = list(CANONICAL_EEG_CHANNELS)
        print(
            f"Canonical EEG channel template (from DOCS / 10-20) has "
            f"{len(self._dataset_channels)} channels: {self._dataset_channels}"
        )

    def _process_file(self, file_info):
        """Read and preprocess a single EDF file, aligning to common channels."""
        path = file_info["path"]
        if not HAS_MNE:
            return None, None

        if self._dataset_channels is None or len(self._dataset_channels) == 0:
            raise RuntimeError(
                "Common channel template (self._dataset_channels) is not set. "
                "Call _collect_common_channels() before processing files."
            )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

            # 1) 标准化通道名，并建立 “标准名 -> 原始名” 映射（参考 TUAB）
            original_ch_names = raw.ch_names
            ch_map = {}  # std_name -> original_name
            for ch in original_ch_names:
                clean = ch.upper().replace("EEG ", "").replace("-REF", "").strip()
                std = self.electrode_set.standardize_name(clean)
                if std not in ch_map:  # 避免重复覆盖
                    ch_map[std] = ch

            # 2) 构建 canonical 顺序的数据矩阵：缺失通道用 0 填补
            # 首先检测原始数据的单位并统一转换为 Volts（MNE RawArray 期望 Volts）
            canonical = self._dataset_channels
            n_samples = int(raw.n_times)
            data_full = np.zeros((len(canonical), n_samples), dtype=np.float32)

            # 先提取一个样本通道来检测单位
            sample_data = None
            for std_ch in canonical:
                if std_ch in ch_map:
                    orig_name = ch_map[std_ch]
                    sample_data = raw.get_data(picks=[orig_name])[0]
                    break
            
            # 检测单位并确定转换因子（转换为 Volts）
            if sample_data is not None:
                max_abs = np.abs(sample_data).max()
                if max_abs < 1.0:  # likely already Volts
                    to_volts_factor = 1.0
                elif max_abs < 1000.0:  # likely mV
                    to_volts_factor = 1e-3
                else:  # likely µV
                    to_volts_factor = 1e-6
            else:
                to_volts_factor = 1.0  # fallback

            for i, std_ch in enumerate(canonical):
                if std_ch in ch_map:
                    orig_name = ch_map[std_ch]
                    # 转换为 Volts
                    data_full[i] = (raw.get_data(picks=[orig_name])[0] * to_volts_factor).astype(
                        np.float32
                    )
                else:
                    # 缺失通道保持为 0，并在日志中提示一次
                    # （不再整条记录跳过）
                    pass

            # 3) 使用 canonical 顺序创建一个新的 RawArray（仅用于统一后处理）
            # RawArray 期望输入是 Volts
            info = mne.create_info(
                ch_names=canonical, sfreq=raw.info["sfreq"], ch_types="eeg"
            )
            raw = mne.io.RawArray(data_full, info, verbose=False)

            # 5) 预处理：陷波、重采样
            if self.filter_notch > 0:
                try:
                    raw.notch_filter(freqs=self.filter_notch, verbose=False)
                except Exception:
                    pass

            if raw.info["sfreq"] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)

            # 6) 取数据并转为 µV（MNE RawArray 返回的是 Volts）
            data = raw.get_data() * 1e6  # Volts -> µV

            if self.clip_threshold is not None:
                data = np.clip(data, -self.clip_threshold, self.clip_threshold)

            # 此时 data 的通道顺序 == self._dataset_channels
            return data, list(raw.ch_names)

        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Seizure annotations: load per-file events from CSV / CSV_BI
    # ------------------------------------------------------------------

    def _load_seizure_intervals(self, edf_path: Path) -> list[tuple[float, float]]:
        """
        Load seizure intervals (start, stop in seconds) from TUH annotations.

        Priority:
        1) *.csv_bi: term-based annotations with labels {bckg,seiz}
        2) *.csv   : event-based annotations; we treat any non-bckg label as seizure

        Returns:
            List of (start_time, stop_time) for seizure periods.
        """
        base = edf_path.with_suffix("")  # strip .edf
        candidates = [
            base.with_suffix(".csv_bi"),
            base.with_suffix(".csv"),
        ]

        intervals: list[tuple[float, float]] = []

        for p in candidates:
            if not p.exists():
                continue
            try:
                with p.open("r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or row[0].startswith("#") or row[0] == "channel":
                            continue
                        # channel,start_time,stop_time,label,confidence
                        if len(row) < 5:
                            continue
                        channel, start, stop, label, conf = row
                        lab = label.strip().lower()
                        if p.suffix == ".csv_bi":
                            # bi-level: 'bckg' or 'seiz'
                            if lab.startswith("seiz"):
                                intervals.append((float(start), float(stop)))
                        else:
                            # full csv: treat any non-background event as seizure
                            if lab != "bckg":
                                intervals.append((float(start), float(stop)))
                # If we successfully loaded any intervals from this file, stop searching
                if intervals:
                    break
            except Exception as e:
                print(f"  Warning: failed to read annotations {p.name}: {e}")
                continue

        # Merge overlapping intervals for robustness
        if not intervals:
            return []
        intervals.sort()
        merged: list[tuple[float, float]] = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        return merged

    def _segment_label(
        self,
        seg_start: float,
        seg_end: float,
        seiz_intervals: list[tuple[float, float]],
        overlap_thresh: float = 0.5,
    ) -> int:
        """
        Determine segment label based on overlap with seizure intervals.

        Args:
            seg_start: segment start time (seconds, relative to file start)
            seg_end: segment end time (seconds)
            seiz_intervals: list of (start, stop) seizure intervals in seconds
            overlap_thresh: minimum fraction of segment duration that must
                            overlap with seizure intervals to be labeled 1

        Returns:
            1 if segment is seizure, 0 otherwise.
        """
        dur = seg_end - seg_start
        if dur <= 0 or not seiz_intervals:
            return 0

        overlap = 0.0
        for s, e in seiz_intervals:
            inter_start = max(seg_start, s)
            inter_end = min(seg_end, e)
            if inter_end > inter_start:
                overlap += inter_end - inter_start

        return 1 if (overlap / dur) >= overlap_thresh else 0

    def _save_dataset_info(self, stats: dict):
        """Save dataset info and processing parameters to JSON."""
        # Use actual channels from data if available; fall back to standard 10-20 set
        channels = (
            self._dataset_channels
            if self._dataset_channels
            else ElectrodeSet.Standard_10_20
        )
        
        info = {
            "dataset": {
                "name": TUEP_INFO.dataset_name,
                "description": "Temple University Hospital EEG Epilepsy Corpus",
                "task_type": str(TUEP_INFO.task_type.value),
                "downstream_task": str(TUEP_INFO.downstream_task_type.value),
                "num_labels": TUEP_INFO.num_labels,
                "category_list": TUEP_INFO.category_list,
                "original_sampling_rate": None,  # TUEP files have variable sampling rates
                "channels": channels,
                "channel_count": len(channels),
                "montage": TUEP_INFO.montage,
                "source_url": "https://www.isip.piconepress.com/projects/tuh_eeg/",
            },
            "processing": {
                "target_sampling_rate": self.target_sfreq,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "filter_notch": self.filter_notch,
                "max_amplitude_uv": self.max_amplitude_uv,
                "clip_threshold": self.clip_threshold,
            },
            "statistics": stats,
            "generated_at": datetime.now().isoformat(),
        }
        
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory path exists as a file: {self.output_dir}. "
                f"Please remove it or choose a different output directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = self.output_dir / "dataset_info.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info to {json_path}")

    def build(self):
        """Build the dataset."""
        if not HAS_MNE:
            raise ImportError("MNE required")
        
        # Reset statistics
        self.total_segments = 0
        self.valid_segments = 0
        self.rejected_segments = 0
        self._dataset_channels = None
            
        subject_map = self._find_files()
        print(f"Found {len(subject_map)} subjects.")

        # 第一次遍历：统计“所有文件共同拥有”的常见标准通道（交集），用于统一对齐
        self._collect_common_channels(subject_map)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_subjects = []
        failed_subjects = []
        
        for sub_id, files in subject_map.items():
            # Determine subject split (assume all files for a subject are in same split)
            split = files[0]["split"]
            label_str = files[0]["label"] # Assume subject label consistency
            
            # Create HDF5
            out_path = self.output_dir / f"sub_{sub_id}.h5"
            # Overwrite existing files to ensure parameters (like window size) are updated
            if out_path.exists():
                print(f"Overwriting existing {sub_id}...")
                out_path.unlink()
                
            print(f"Processing Subject {sub_id} ({split}, {label_str})...")
            
            # Process first file to get channel names before creating HDF5Writer
            ch_names = None
            subject_attrs = None
            first_valid_file_data = None
            first_valid_file_info = None
            # Per-subject label statistics
            label_counts = {0: 0, 1: 0}  # 0=non_seizure, 1=seizure
            # Per-subject total segment counter (across all EDF files)
            subject_total_segments = 0
            
            try:
                # Sort files by session and trial to ensure deterministic order
                files.sort(key=lambda x: x["stem"])
                
                # Find first valid file to get channel names (already aligned to canonical)
                for f_info in files:
                    data, processed_ch_names = self._process_file(f_info)
                    if data is not None and processed_ch_names is not None:
                        ch_names = processed_ch_names
                        first_valid_file_data = data
                        first_valid_file_info = f_info
                        # Store channel names for dataset info (first subject sets it)
                        if self._dataset_channels is None:
                            self._dataset_channels = ch_names
                        break
                
                # If no valid file found, skip this subject
                if ch_names is None:
                    print(f"  Subject {sub_id}: No valid files found, skipping")
                    failed_subjects.append(sub_id)
                    continue
                
                # Create subject attributes with 全局统一通道名（common channels）
                chn_name = self._dataset_channels if self._dataset_channels else ch_names
                subject_attrs = SubjectAttrs(
                    subject_id=sub_id,
                    dataset_name=f"TUEP_{split}",
                    task_type="seizure",
                    downstream_task_type="classification",
                    rsFreq=self.target_sfreq,
                    chn_name=chn_name,
                    num_labels=2,
                    category_list=["non_seizure", "seizure"],
                    chn_pos=None,
                    chn_ori=None,
                    chn_type="EEG",
                    montage="10_20"
                )
                
                # Now create HDF5Writer with proper subject_attrs
                with HDF5Writer(str(out_path), subject_attrs) as writer:
                    # Process all files (including the first one we already processed)
                    for unique_trial_id, f_info in enumerate(files):
                        # Use cached data for first valid file, otherwise process
                        if f_info["path"] == first_valid_file_info["path"]:
                            data = first_valid_file_data
                        else:
                            data, _ = self._process_file(f_info)
                            if data is None:
                                continue
                        
                        # Trial attributes
                        # Use filename as trial info
                        trial_id_match = re.search(r't(\d+)', f_info["stem"])
                        original_trial_num = int(trial_id_match.group(1)) if trial_id_match else 0
                        
                        sess_id_match = re.search(r's(\d+)', f_info["stem"])
                        sess_num = int(sess_id_match.group(1)) if sess_id_match else 0
                        
                        trial_attrs = TrialAttrs(
                            trial_id=unique_trial_id, # Use unique incremental ID to avoid collisions
                            session_id=sess_num,
                            task_name=f_info["label"]
                        )
                        trial_name = writer.add_trial(trial_attrs)
                        
                        # Segment
                        n_samples = data.shape[1]
                        seg_idx = 0
                        # Simple directory-based label: entire file gets same label
                        # 00_epilepsy -> seizure=1, 01_no_epilepsy -> non_seizure=0
                        label_val = 1 if f_info["label"] == "seizure" else 0
                        
                        for start in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                            end = start + self.window_samples
                            segment = data[:, start:end]

                            self.total_segments += 1

                            # Validate Amplitude
                            if self.max_amplitude_uv is not None and np.abs(segment).max() > self.max_amplitude_uv:
                                self.rejected_segments += 1
                                continue

                            self.valid_segments += 1
                            # Use file-level label for all segments (based on directory)
                            label_counts[label_val] += 1

                            seg_attrs = SegmentAttrs(
                                segment_id=seg_idx,
                                start_time=start / self.target_sfreq,
                                end_time=end / self.target_sfreq,
                                time_length=self.window_sec,
                                label=np.array([label_val])
                            )
                            writer.add_segment(trial_name, seg_attrs, segment)
                            seg_idx += 1

                        # Accumulate per-file segments into per-subject total
                        subject_total_segments += seg_idx
                
                successful_subjects.append(sub_id)
                print(f"  Subject {sub_id}: {subject_total_segments} segments")
                # Print per-subject label distribution
                total_subj_segs = label_counts[0] + label_counts[1]
                if total_subj_segs > 0:
                    pct_non = label_counts[0] / total_subj_segs * 100
                    pct_seiz = label_counts[1] / total_subj_segs * 100
                    print(
                        f"    Label distribution: "
                        f"non_seizure= {label_counts[0]} ({pct_non:.1f}%), "
                        f"seizure= {label_counts[1]} ({pct_seiz:.1f}%)"
                    )
                    
            except Exception as e:
                print(f"Error processing subject {sub_id}: {e}")
                failed_subjects.append(sub_id)
                import traceback
                traceback.print_exc()
        
        # Summary report
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total subjects: {len(subject_map)}")
        print(f"Successful: {len(successful_subjects)}")
        print(f"Failed: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed subject IDs: {failed_subjects[:20]}{'...' if len(failed_subjects) > 20 else ''}")
        print(f"\nTotal segments: {self.total_segments}")
        print(f"Valid segments: {self.valid_segments}")
        print(f"Rejected segments: {self.rejected_segments}")
        if self.total_segments > 0:
            print(f"Rejection rate: {self.rejected_segments / self.total_segments * 100:.1f}%")
        print("=" * 50)
        
        # Save dataset info JSON
        stats = {
            "total_subjects": len(subject_map),
            "successful_subjects": len(successful_subjects),
            "failed_subjects": failed_subjects,
            "total_segments": self.total_segments,
            "valid_segments": self.valid_segments,
            "rejected_segments": self.rejected_segments,
            "rejection_rate": f"{self.rejected_segments / self.total_segments * 100:.1f}%" if self.total_segments > 0 else "0%",
        }
        self._save_dataset_info(stats)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TUEP HDF5 dataset")
    parser.add_argument("raw_data_dir", help="Directory containing raw files (should have train/eval with seizure/non_seizure subdirectories)")
    parser.add_argument("--output_dir", default="./hdf5", help="Output directory")
    parser.add_argument("--target_sfreq", type=float, default=200.0, help="Target sampling rate (Hz)")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=10.0, help="Stride length in seconds")
    parser.add_argument("--filter_notch", type=float, default=60.0, help="Notch filter frequency (Hz, 0 to disable)")
    parser.add_argument("--max_amplitude_uv", type=float, default=None, help="Maximum amplitude threshold (µV, None to disable)")
    parser.add_argument("--clip_threshold", type=float, default=None, help="Amplitude clipping threshold (µV, None to disable)")
    parser.add_argument(
        "--tcp_mode",
        type=str,
        default=None,
        choices=["01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a"],
        help="Only process files from this TCP montage (matches directory name, e.g. 02_tcp_le). "
             "If omitted, all montages are processed together.",
    )
    args = parser.parse_args()

    builder = TUEPBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
        clip_threshold=args.clip_threshold,
        tcp_mode=args.tcp_mode,
    )
    builder.build()
