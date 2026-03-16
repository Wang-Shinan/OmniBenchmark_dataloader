import numpy as np
import h5py
from pathlib import Path
import scipy.io
import pandas as pd

# ======================= 路径配置 =======================
RAW_ROOT = Path("/mnt/dataset2/Datasets/EAV/EAV")
OUT_DIR = Path("/mnt/dataset2/benchmark_hdf5/EAV_5class_emoeeg_style")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= 参数配置 =======================
DATASET_NAME = "EAV"
N_SUBJECTS = 42
N_TRIALS = 100

SFREQ = 200.0
SEG_LEN_SEC = 2.0
SEG_LEN = int(SFREQ * SEG_LEN_SEC)   # 400
STRIDE = SEG_LEN                     # non-overlap

LABEL_AS_ONEHOT = True
N_CLASS = 5                          # EAV 五类情绪

# ======================= 通道名（30 ch） =======================
CH_NAMES = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz',
    'FC1','FC2','CP1','CP2','FC5','FC6','CP5','CP6',
    'TP9','TP10'
]
CH_NAMES_H5 = np.array(CH_NAMES, dtype="S")
assert len(CH_NAMES) == 30

# ======================= 小工具 =======================
def onehot(y, k):
    v = np.zeros((k,), dtype=np.uint8)
    v[int(y)] = 1
    return v

# ======================= 读取单个 subject =======================
def load_subject(subject_dir: Path):
    """
    返回：
      seg  : (10000, 30, 200) or (T, C, L) → reshape later
      label: (5, 100) one-hot
    """
    eeg_mat = subject_dir / "EEG" / f"{subject_dir.name}_eeg.mat"
    lab_mat = subject_dir / "EEG" / f"{subject_dir.name}_eeg_label.mat"

    eeg = scipy.io.loadmat(eeg_mat)
    lab = scipy.io.loadmat(lab_mat)

    # 你之前已经确认：seg / label
    seg = eeg["seg"]          # (10000, 30, 200)
    label = lab["label"]      # (5, 100)

    return seg, label

# ======================= 主 builder =======================
def build_one_subject(sid: int):
    subject_dir = RAW_ROOT / f"subject{sid}"
    out_path = OUT_DIR / f"sub_{sid}.h5"

    if out_path.exists():
        print(f"[SKIP] exists: {out_path}")
        return

    seg, label = load_subject(subject_dir)
    print(f"[INFO] subject {sid}: seg={seg.shape}, label={label.shape}")

    with h5py.File(out_path, "w") as f:
        for trial_id in range(N_TRIALS):
            trial_grp = f.create_group(f"trial{trial_id}")

            # label: one-hot(5,)
            y = label[:, trial_id]
            if LABEL_AS_ONEHOT:
                y_store = y.astype(np.uint8)
            else:
                y_store = int(np.argmax(y))

            # trial data: (T, C, L)
            trial_data = seg[trial_id * 100:(trial_id + 1) * 100]  # 100 segments
            trial_data = np.transpose(trial_data, (1, 0, 2))      # (C, 100, 200)
            trial_data = trial_data.reshape(30, -1)               # (30, 20000)

            n_samples = trial_data.shape[1]

            seg_id = 0
            for start in range(0, n_samples - SEG_LEN + 1, STRIDE):
                end = start + SEG_LEN
                slice_data = trial_data[:, start:end]             # (30, 400)

                samp_grp = trial_grp.create_group(f"sample{seg_id}")
                dset = samp_grp.create_dataset("eeg", data=slice_data)

                # ========== attrs（emoeeg 对齐） ==========
                dset.attrs["rsFreq"] = int(SFREQ)
                dset.attrs["time_length"] = float(SEG_LEN_SEC)
                dset.attrs["label"] = y_store

                dset.attrs["subject_id"] = sid
                dset.attrs["trial_id"] = trial_id
                dset.attrs["session_id"] = 0
                dset.attrs["segment_id"] = seg_id
                dset.attrs["start_sample"] = int(start)

                dset.attrs["dataset_name"] = DATASET_NAME
                dset.attrs["chn_name"] = CH_NAMES_H5
                dset.attrs["chn_pos"] = "None"
                dset.attrs["chn_ori"] = "None"
                dset.attrs["chn_type"] = "EEG"
                dset.attrs["montage"] = "10_20"

                seg_id += 1

    print(f"[OK] saved: {out_path}")

# ======================= main =======================
def main():
    for sid in range(1, N_SUBJECTS + 1):
        build_one_subject(sid)

if __name__ == "__main__":
    main()