import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from datasets.chinese_eeg import ChineseEEGBuilder
from schema import DatasetInfo, PreprocessingConfig

# -----------------------------------------------------------
# 辅助：生成 EGI 128 标准通道名
# -----------------------------------------------------------
def get_egi_channels():
    # EGI 通道名为 E1, E2, ..., E128
    return [f"E{i}" for i in range(1, 129)]

# -----------------------------------------------------------
# JSON 生成函数
# -----------------------------------------------------------
def generate_dataset_info_json(output_dir, book_name, total_subs, total_segs):
    """生成符合规范的 JSON 文件"""
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "dataset": {
            "name": f"ChineseEEG_{book_name}",
            "task_type": "reading_detection",
            "downstream_task": "classification",
            "num_labels": 2,
            "category_list": ["No-Reading", "Reading"],
            "original_sampling_rate": 1000.0,
            # ✅ 通道名：E1 - E128
            "channels": [f"E{i}" for i in range(1, 129)],
            "montage": "GSN-HydroCel-128"
        },
        "processing": {
            "target_sampling_rate": 200.0,
            "window_sec": 1.0,
            "stride_sec": 1.0,
            "filter_low": 0.1,
            "filter_high": 75.0,
            "filter_notch": 50.0,
            "max_amplitude_uv": 600.0,
            "units_stored": "uV",
            "notes": "Processed from ChineseEEG dataset (EGI 128). VAD aligned labels. Greedy non-overlapping sampling to maximize yield."
        },
        "segmentation": {
            "policy": "greedy_non_overlapping_quota",
            "window_sec": 1.0,
            "stride_sec": "variable (greedy)",
            "event_source": "VAD (Voice Activity Detection) majority vote",
            "notes": "Max 40 segments per class per run. Scanned with 0.1s stride, jumped 1.0s upon capture."
        },
        "statistics": {
            "total_subjects": total_subs,
            "successful": total_subs,
            "failed": 0,
            "total_segments": total_segs,
            "valid_segments": total_segs,
            "rejected_segments": 0
        },
        "generated_at": datetime.now().isoformat()
    }
    
    json_path = output_dir / "dataset_info.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"   📝 JSON Generated: {json_path}")

# -----------------------------------------------------------
# 主执行逻辑
# -----------------------------------------------------------
def main():
    possible_paths = [
        r"Z:\Processed_datasets\EEG_Bench\ChineseEEG2",
        r"Z:\Processed_datasets\EEG_Bench\ChineseEEG2",
        os.path.abspath("ChineseEEG2")
    ]
    raw_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            raw_dir = Path(p)
            break
            
    if not raw_dir:
        print("❌ Error: Could not find 'ChineseEEG2' directory.")
        return
    print(f"📂 Source Data: {raw_dir}")

    tasks = [
        ("garnettdream", "ChineseEEG_garnettdream"),
        ("littleprince", "ChineseEEG_littleprince")
    ]
    
    base_out_dir = Path(r"Z:\Processed_datasets\EEG_Bench\ChineseEEG2\last") 
    if not os.path.exists(r"Z:\Processed_datasets"):
        base_out_dir = Path(r"D:\Processed_datasets\EEG_Bench")
        
    for book_name, folder_name in tasks:
        target_dir = base_out_dir / folder_name
        
        print(f"\n{'#'*60}")
        print(f"📘 Processing Book: {book_name}")
        print(f"📂 Output Folder: {target_dir}")
        print(f"{'#'*60}")
        
        ds_info = DatasetInfo(dataset_name=f"ChineseEEG_{book_name}", original_sampling_rate=1000)
        pre_conf = PreprocessingConfig(target_sampling_rate=200, output_dir=str(target_dir))
        
        builder = ChineseEEGBuilder(raw_dir, target_dir, ds_info, pre_conf, book_name=book_name)
        
        sub_ids = builder.get_subject_ids()
        if not sub_ids:
             print("   ⚠️ No subjects found.")
             generate_dataset_info_json(target_dir, book_name, 0, 0)
             continue

        total_segments_book = 0
        for sub_id in sub_ids:
            n_segs = builder.build_subject(sub_id)
            total_segments_book += n_segs
            
        generate_dataset_info_json(target_dir, book_name, len(sub_ids), total_segments_book)

if __name__ == "__main__":
    main()