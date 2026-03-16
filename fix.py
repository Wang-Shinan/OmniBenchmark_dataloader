from pathlib import Path
import h5py

def patch_h5_add_chnpos(root_dir: str):
    root = Path(root_dir)
    h5_files = sorted(root.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files in {root_dir}")
    for p in h5_files:
        print(f"  Patching {p.name} ...", end=" ")
        with h5py.File(p, "r+") as f:
            # 如果没有 chn_pos / chn_ori，就补一个 "None"
            if "chn_pos" not in f.attrs:
                f.attrs["chn_pos"] = "None"
            if "chn_ori" not in f.attrs:
                f.attrs["chn_ori"] = "None"
        print("done.")

# 1) 修补 ChineseEEG2 last 目录
patch_h5_add_chnpos("/mnt/dataset2/Processed_datasets/EEG_Bench/ChineseEEG2/last")

# 如需要，也可以顺手把旧 ChineseEEG 数据集一起补上
# patch_h5_add_chnpos("/mnt/dataset2/Processed_datasets/EEG_Bench/ChineseEEG_garnettdream")