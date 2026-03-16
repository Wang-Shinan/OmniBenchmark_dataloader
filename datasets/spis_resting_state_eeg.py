#!/usr/bin/env python3
"""
RestingStateEEG (Pre-SART Resting, Eyes-Open/Eyes-Closed) Builder.

Dataset:
- Name: RestingStateEEG (a.k.a. "Resting state EEG", pre-SART resting recordings)
- Subjects: 10 healthy participants (S02-S11 in the provided file list)
- Conditions: Eyes Open (EO) and Eyes Closed (EC), typically collected sequentially
- Modality: EEG (BioSemi ActiveTwo, 64 channels, 10-10 system; MAT exports commonly at 256 Hz)

Expected raw directory layout (flexible):
- RAW_DATA_DIR/
    Pre-SART EEG/
        S02_restingPre_EO.mat
        S02_restingPre_EC.mat
        ...
  or RAW_DATA_DIR/ containing the above *.mat files directly.

Segmentation strategy (policy: "rest_sliding_window"):
- This is continuous resting-state EEG without discrete stimulus events.
- We therefore segment with fixed-length sliding windows across the full recording.
- Default window/stride are group-standard (1s/1s) but NOT locked: CLI allows override.

Labeling strategy:
- Supervised condition labels from filenames:
    EO -> 0
    EC -> 1

QC rules (minimum, per v1.4):
- Reject segments containing NaN/Inf.
- Reject segments whose max absolute amplitude exceeds a threshold (in microvolts, µV).
  Default: 600 µV (CLI configurable).

Outputs:
- One HDF5 per subject: OUTPUT_DIR/RestingStateEEG/sub_<subject_id>.h5
  - 2 trials per subject: trial0=EO, trial1=EC
  - segments stored as [C, T] arrays in µV
- One dataset-level metadata JSON:
    OUTPUT_DIR/RestingStateEEG/dataset_info.json

CLI usage:
    python -m benchmark_dataloader.datasets.resting_state_eeg RAW_DATA_DIR --output_dir OUTPUT_DIR
    # or
    python resting_state_eeg.py RAW_DATA_DIR --output_dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---- project-compatible imports (preferred), with standalone fallback ----
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType
except Exception:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType


# =========================
# Mandatory constants (v1.4)
# =========================
BUILDER_VERSION = "resting_state_eeg_builder_v1.2"

# Canonical 64-channel list (from common 10-10/10-20 naming). Used as fallback if MAT lacks names.
CHANNELS_64 = [
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
]

# Label mapping
LABEL_MAP = {"EO": 0, "EC": 1}
CATEGORY_LIST = ["EO", "EC"]

RESTING_STATE_EEG_INFO = DatasetInfo(
    dataset_name="RestingStateEEG",
    task_type=DatasetTaskType.RESTING_STATE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=2,
    category_list=CATEGORY_LIST,
    sampling_rate=256.0,  # typical MAT export; final rsFreq is target_sfreq
    montage="standard_1010",
    channels=CHANNELS_64,
)


# =========================
# Helpers
# =========================
_FILE_RE = re.compile(r"(?P<sub>S\d+)_restingPre_(?P<cond>EO|EC)\.mat$", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_mat_files(raw_data_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Return mapping: subject_id -> { 'EO': path, 'EC': path }."""
    candidates: List[Path] = []
    # prefer expected folder, but allow anywhere under raw_data_dir
    pre_sart = raw_data_dir / "Pre-SART EEG"
    if pre_sart.exists() and pre_sart.is_dir():
        candidates.extend(sorted(pre_sart.rglob("*.mat")))
    candidates.extend(sorted(raw_data_dir.rglob("*.mat")))

    subj_map: Dict[str, Dict[str, Path]] = {}
    for p in candidates:
        m = _FILE_RE.search(p.name)
        if not m:
            continue
        sub = m.group("sub").upper()
        cond = m.group("cond").upper()
        subj_map.setdefault(sub, {})[cond] = p
    return subj_map


def _infer_unit_and_to_volts(x: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    Infer unit by robust amplitude and convert to Volts for MNE.

    Notes:
    - Different releases / exports of this dataset may store EEG samples in different scales
      (e.g., V / mV / µV / nV / pV). Some MAT exports appear to be in **pV** (picovolts),
      where typical EEG magnitudes (~10–100 µV) correspond to ~1e7–1e8 in raw numbers.
    - We use a conservative heuristic based on P99(|x|) and map to the most plausible unit.

    Heuristic (P99 of absolute values):
      p99 < 1e-2        -> V
      1e-2  <= p99 < 1  -> mV
      1     <= p99 < 1e4 -> µV
      1e4   <= p99 < 1e7 -> nV
      1e7   <= p99       -> pV

    Returns:
      (x_in_volts, inferred_unit, p99_abs)
    """
    x = np.asarray(x)
    p99 = float(np.nanpercentile(np.abs(x), 99))

    if p99 < 1e-2:
        return x.astype(np.float64, copy=False), "V", p99
    if p99 < 1:
        return (x * 1e-3).astype(np.float64, copy=False), "mV", p99
    if p99 < 1e4:
        return (x * 1e-6).astype(np.float64, copy=False), "uV", p99
    if p99 < 1e7:
        return (x * 1e-9).astype(np.float64, copy=False), "nV", p99
    return (x * 1e-12).astype(np.float64, copy=False), "pV", p99
def _extract_data_and_meta(mat: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[str]], Optional[float]]:
    """
    Try to extract EEG array, channel names, and sampling rate from a loaded .mat dict.

    Returns:
        data: np.ndarray shape (n_channels, n_samples) in *original numeric unit*
        ch_names: list[str] or None
        sfreq: float or None
    """
    # 1) Sampling rate keys (best-effort)
    sfreq = None
    for k in ["fs", "srate", "sfreq", "sampling_rate", "samplingRate", "Fs", "Srate"]:
        if k in mat:
            try:
                val = float(np.squeeze(mat[k]))
                if np.isfinite(val) and val > 0:
                    sfreq = val
                    break
            except Exception:
                pass

    # 2) Channel names keys (best-effort)
    ch_names: Optional[List[str]] = None
    for k in ["channelList", "chanlabels", "ch_names", "labels", "channels"]:
        if k in mat:
            try:
                arr = mat[k]
                # handle matlab cell arrays / nested arrays
                flat = np.ravel(arr)
                names: List[str] = []
                for item in flat:
                    if isinstance(item, bytes):
                        names.append(item.decode(errors="ignore"))
                    elif isinstance(item, str):
                        names.append(item)
                    else:
                        # matlab strings sometimes appear as ndarray of chars
                        try:
                            s = "".join(chr(c) for c in np.ravel(item).tolist() if isinstance(c, (int, np.integer)))
                            if s:
                                names.append(s)
                        except Exception:
                            pass
                names = [n.strip().upper() for n in names if n and str(n).strip()]
                if len(names) >= 2:
                    ch_names = names
                    break
            except Exception:
                pass

    # 3) EEG data array: find a numeric array that looks like EEG
    # Priority:
    #   (a) arrays that contain a 64-channel dimension (64 × T, T × 64, or E × 64 × T)
    #   (b) otherwise, largest 2D numeric array; fallback to largest numeric array
    data_candidates: List[Tuple[str, np.ndarray]] = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.ndim in (2, 3):
            if v.size < 1000:
                continue
            data_candidates.append((k, v))

    if not data_candidates:
        raise RuntimeError("No numeric EEG-like array found in .mat file keys.")

    def _has_64ch(a: np.ndarray) -> bool:
        return 64 in a.shape

    # Prefer candidates that include a 64-channel dimension
    candidates_64 = [(k, v) for (k, v) in data_candidates if _has_64ch(v)]
    if candidates_64:
        # Prefer 2D first (cleanest), then by size
        data_key, arr = sorted(candidates_64, key=lambda kv: (kv[1].ndim, -kv[1].size))[0]
    else:
        # Prefer 2D first, then by size
        data_key, arr = sorted(data_candidates, key=lambda kv: (kv[1].ndim, -kv[1].size))[0]

    if arr.ndim == 3:
        # common pattern: (n_epochs, n_channels, n_samples) or (n_channels, n_samples, n_epochs)
        # We convert to continuous by concatenating along time.
        a = arr
        # try common orders
        if a.shape[1] <= 256 and a.shape[2] >= 50:  # (E,C,T)
            a = np.transpose(a, (1, 0, 2))  # (C,E,T)
            a = a.reshape(a.shape[0], -1)
        elif a.shape[0] <= 256 and a.shape[1] >= 50:  # (C,T,E) or (C,E,T) ambiguous
            # if last dim is epochs small, assume (C,T,E)
            if a.shape[2] < 2000:
                a = a.reshape(a.shape[0], -1)
            else:
                a = np.transpose(a, (0, 1, 2)).reshape(a.shape[0], -1)
        else:
            # generic: move smallest dim to channels
            ch_dim = int(np.argmin(a.shape))
            a = np.moveaxis(a, ch_dim, 0)
            a = a.reshape(a.shape[0], -1)
        arr2d = a
    else:
        arr2d = arr

    # Ensure shape (C, T)
    if arr2d.shape[0] > arr2d.shape[1] and arr2d.shape[1] <= 256:
        # likely (T, C) where C small
        arr2d = arr2d.T

    # If channels still look wrong, try transpose if it helps match ch_names length
    if ch_names is not None and arr2d.shape[0] != len(ch_names) and arr2d.shape[1] == len(ch_names):
        arr2d = arr2d.T

    data = np.asarray(arr2d, dtype=np.float64)

    # sanitize non-finite early
    # (QC later is per-segment; here just keep as-is but ensure array is float)
    return data, ch_names, sfreq


def _make_mne_raw(data_v: np.ndarray, ch_names: List[str], sfreq: float) -> "mne.io.RawArray":
    """Create MNE RawArray from data in Volts."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(data_v, info, verbose=False)
    # montage is optional; if names match standard montage, set it
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    except Exception:
        pass
    return raw


def _segment_fixed_windows(
    raw: "mne.io.BaseRaw",
    window_sec: float,
    stride_sec: float,
) -> np.ndarray:
    """Return epochs data as (n_epochs, n_ch, n_t) in Volts."""
    overlap = max(window_sec - stride_sec, 0.0)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(window_sec),
        overlap=float(overlap),
        preload=True,
        verbose=False,
    )
    return epochs.get_data()  # (n_epochs, n_ch, n_t) in V


# =========================
# Builder
# =========================
class RestingStateEEGBuilder:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str = "./hdf5_datasets",
        subjects: Optional[List[str]] = None,
        # unified preprocessing params (defaults are group-standard, but not locked)
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 1.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        # QC
        max_amplitude_uv: float = 600.0,
    ):
        if not HAS_MNE:
            raise ImportError("MNE is required for RestingStateEEGBuilder")
        if not HAS_SCIPY:
            raise ImportError("scipy is required to load .mat files for RestingStateEEGBuilder")

        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir) / RESTING_STATE_EEG_INFO.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.subjects_filter = [s.upper() for s in subjects] if subjects else None

        self.target_sfreq = float(target_sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.filter_low = float(filter_low)
        self.filter_high = float(filter_high)
        self.filter_notch = float(filter_notch)
        self.max_amplitude_uv = float(max_amplitude_uv)

        self.subj_files = _find_mat_files(self.raw_data_dir)

    def get_subject_ids(self) -> List[str]:
        subs = sorted(self.subj_files.keys())
        if self.subjects_filter:
            subs = [s for s in subs if s in set(self.subjects_filter)]
        return subs

    def build_subject(self, subject_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Build one subject and return:
        - output file path
        - per-subject summary dict (segments/QC/unit inference)
        """
        subject_id = subject_id.upper()
        if subject_id not in self.subj_files:
            raise ValueError(f"Unknown subject_id: {subject_id}")

        cond_map = self.subj_files[subject_id]
        missing = [c for c in ["EO", "EC"] if c not in cond_map]
        if missing:
            raise FileNotFoundError(f"Subject {subject_id} missing conditions: {missing}")

        out_file = self.output_dir / f"sub_{subject_id}.h5"
        # allow overwrite to ensure determinism across runs
        if out_file.exists():
            out_file.unlink()

        # per-condition processing
        cond_data_uv: Dict[str, np.ndarray] = {}
        cond_meta: Dict[str, Dict[str, Any]] = {}

        for cond in ["EO", "EC"]:
            mat_path = cond_map[cond]
            mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            data, ch_names, sfreq = _extract_data_and_meta(mat)

            # Some MAT exports include extra non-EEG channels (e.g., EXG/status),
            # commonly resulting in 68 channels total. For benchmark consistency,
            # we keep the first 64 EEG channels when a canonical 64 list is available.
            if data.ndim == 2 and data.shape[0] > 64 and data.shape[0] in (67, 68) and len(CHANNELS_64) == 64:
                data = data[:64, :]
                if ch_names is not None and len(ch_names) >= 64:
                    ch_names = [str(n).strip().upper() for n in ch_names[:64]]

            # choose channel names fallback
            if ch_names is None or len(ch_names) != data.shape[0]:
                # fallback to canonical list if sizes match
                if data.shape[0] == len(CHANNELS_64):
                    ch_names_use = CHANNELS_64
                else:
                    # generic fallback
                    ch_names_use = [f"CH{idx+1:03d}" for idx in range(data.shape[0])]
            else:
                ch_names_use = ch_names

            sfreq_use = float(sfreq) if (sfreq is not None and sfreq > 0) else float(RESTING_STATE_EEG_INFO.sampling_rate)

            data_v, inferred_unit, p99 = _infer_unit_and_to_volts(data)

            raw = _make_mne_raw(data_v, ch_names_use, sfreq_use)

            # preprocess
            # notch first or after bandpass? both acceptable; keep consistent with other builders: bandpass then notch.
            raw.filter(l_freq=self.filter_low, h_freq=self.filter_high, verbose=False)
            if self.filter_notch and self.filter_notch > 0:
                raw.notch_filter(freqs=[self.filter_notch], verbose=False)
            if raw.info["sfreq"] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)

            X_v = _segment_fixed_windows(raw, self.window_sec, self.stride_sec)  # (E,C,T) in V
            X_uv = X_v * 1e6  # store in µV

            # amplitude diagnostics after preprocessing (µV)
            amp_p99_uv = float(np.nanpercentile(np.abs(X_uv), 99))
            amp_max_uv = float(np.nanmax(np.abs(X_uv)))

            cond_data_uv[cond] = X_uv
            cond_meta[cond] = {
                "mat_path": str(mat_path),
                "orig_sfreq": sfreq_use,
                "post_sfreq": float(raw.info["sfreq"]),
                "inferred_unit": inferred_unit,
                "p99_abs": p99,
                "n_channels": int(X_uv.shape[1]),
                "n_samples_per_seg": int(X_uv.shape[2]),
                "n_segments_total": int(X_uv.shape[0]),
                "amp_p99_uv_post": amp_p99_uv,
                "amp_max_uv_post": amp_max_uv,
                "ch_names": ch_names_use,
                "data_key_hint": None,  # could be added if needed
            }

        # subject attrs (post-resample)
                # quick diagnostics (helps detect unit/array selection issues)
        try:
            eo_amp = cond_meta["EO"]["amp_p99_uv_post"]
            eo_max = cond_meta["EO"]["amp_max_uv_post"]
            ec_amp = cond_meta["EC"]["amp_p99_uv_post"]
            ec_max = cond_meta["EC"]["amp_max_uv_post"]
            print(f"    [diag] {subject_id} amp_p99_uv_post EO/EC = {eo_amp:.2f}/{ec_amp:.2f}, amp_max_uv_post EO/EC = {eo_max:.2f}/{ec_max:.2f}")
        except Exception:
            pass

        any_cond = "EO"
        ch_names_final = cond_meta[any_cond]["ch_names"]
        subject_attrs = SubjectAttrs(
            subject_id=subject_id,
            dataset_name=RESTING_STATE_EEG_INFO.dataset_name,
            task_type=RESTING_STATE_EEG_INFO.task_type.value,
            downstream_task_type=RESTING_STATE_EEG_INFO.downstream_task_type.value,
            rsFreq=self.target_sfreq,
            chn_name=ch_names_final,
            num_labels=RESTING_STATE_EEG_INFO.num_labels,
            category_list=RESTING_STATE_EEG_INFO.category_list,
            chn_type="EEG",
            montage=RESTING_STATE_EEG_INFO.montage,
        )

        # QC + write
        qc_counts = {
            "total_segments": 0,
            "valid_segments": 0,
            "rejected_nonfinite": 0,
            "rejected_amplitude": 0,
        }

        with HDF5Writer(str(out_file), subject_attrs) as writer:
            # Two trials: EO then EC
            trial_specs = [("EO", 0, "rest_EO"), ("EC", 1, "rest_EC")]
            global_seg_id = 0

            for cond, trial_id, task_name in trial_specs:
                trial_attrs = TrialAttrs(trial_id=trial_id, session_id=0, task_name=task_name)
                trial_name = writer.add_trial(trial_attrs)

                X_uv = cond_data_uv[cond]  # (E,C,T) in µV
                y = LABEL_MAP[cond]

                for i in range(X_uv.shape[0]):
                    seg = X_uv[i]
                    qc_counts["total_segments"] += 1

                    if not np.isfinite(seg).all():
                        qc_counts["rejected_nonfinite"] += 1
                        continue

                    if self.max_amplitude_uv and self.max_amplitude_uv > 0:
                        if float(np.max(np.abs(seg))) > self.max_amplitude_uv:
                            qc_counts["rejected_amplitude"] += 1
                            continue

                    start_t = float(i * self.stride_sec)
                    end_t = float(start_t + self.window_sec)

                    seg_attrs = SegmentAttrs(
                        segment_id=global_seg_id,
                        start_time=start_t,
                        end_time=end_t,
                        time_length=self.window_sec,
                        label=np.array([y], dtype=np.int64),
                        task_label=cond,
                    )
                    writer.add_segment(trial_name, seg_attrs, seg.astype(np.float32, copy=False))
                    qc_counts["valid_segments"] += 1
                    global_seg_id += 1

        summary = {
            "subject_id": subject_id,
            "output_file": str(out_file),
            "conditions": cond_meta,
            "qc": qc_counts,
        }
        return str(out_file), summary

    def build_all(self) -> Dict[str, Any]:
        subject_ids = self.get_subject_ids()
        if not subject_ids:
            raise RuntimeError(f"No subjects found under {self.raw_data_dir}")

        print("============================================================")
        print(f"Dataset: {RESTING_STATE_EEG_INFO.dataset_name}")
        print(f"Raw data dir: {self.raw_data_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Subjects: {len(subject_ids)}")
        print(
            f"Preproc: target_sfreq={self.target_sfreq}, window_sec={self.window_sec}, stride_sec={self.stride_sec}, "
            f"filter=[{self.filter_low}-{self.filter_high}]Hz, notch={self.filter_notch}Hz"
        )
        print(f"QC: max_amplitude_uv={self.max_amplitude_uv} µV")
        print("============================================================")

        all_summaries: List[Dict[str, Any]] = []
        total_qc = {
            "total_segments": 0,
            "valid_segments": 0,
            "rejected_nonfinite": 0,
            "rejected_amplitude": 0,
        }

        for idx, sid in enumerate(subject_ids, 1):
            print(f"[{idx}/{len(subject_ids)}] Processing subject: {sid}")
            out_path, summ = self.build_subject(sid)
            all_summaries.append(summ)
            for k in total_qc:
                total_qc[k] += int(summ["qc"][k])
            print(
                f"    segments(valid/total)={summ['qc']['valid_segments']}/{summ['qc']['total_segments']} "
                f"(rej_nonfinite={summ['qc']['rejected_nonfinite']}, rej_amp={summ['qc']['rejected_amplitude']})"
            )
            print(f"    -> {out_path}")

        if total_qc["valid_segments"] <= 0:
            raise RuntimeError(
                "Builder produced 0 valid segments overall. "
                "Check unit inference, max_amplitude_uv, and raw file integrity."
            )

        # write dataset_info.json (v1.4 required)
        dataset_info = self._make_dataset_info_json(subject_ids, all_summaries, total_qc)
        info_path = self.output_dir / "dataset_info.json"
        info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False))
        print(f"[OK] Wrote dataset_info.json -> {info_path}")

        return dataset_info

    def _make_dataset_info_json(
        self,
        subject_ids: List[str],
        all_summaries: List[Dict[str, Any]],
        total_qc: Dict[str, int],
    ) -> Dict[str, Any]:
        # summarize unit inference
        unit_summary = {}
        for summ in all_summaries:
            sid = summ["subject_id"]
            unit_summary[sid] = {
                cond: {
                    "inferred_unit": summ["conditions"][cond]["inferred_unit"],
                    "p99_abs": summ["conditions"][cond]["p99_abs"],
                    "orig_sfreq": summ["conditions"][cond]["orig_sfreq"],
                    "post_sfreq": summ["conditions"][cond]["post_sfreq"],
                    "n_segments_total": summ["conditions"][cond]["n_segments_total"],
                }
                for cond in ["EO", "EC"]
            }

        # v1.4 segmentation declaration
        segmentation = {
            "policy": "rest_sliding_window",
            "parameters": {
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
                "alignment": None,
                "event_source": None,
                "notes": (
                    "Continuous resting-state EEG; segments are fixed-length windows across the full recording. "
                    "EO/EC are stored as separate trials."
                ),
            },
            "rationale": (
                "Resting-state EEG has no discrete stimulus/event markers. "
                "Fixed-length sliding windows provide uniform segments for foundation-model training and downstream EO/EC classification."
            ),
        }

        labeling = {
            "label_policy": "supervised_condition",
            "mapping": {"EO": 0, "EC": 1},
            "num_labels": int(RESTING_STATE_EEG_INFO.num_labels),
            "category_list": list(RESTING_STATE_EEG_INFO.category_list),
            "notes": "Labels are derived deterministically from file condition suffix (_EO/_EC).",
        }

        preprocessing = {
            "target_sfreq": self.target_sfreq,
            "filter_low": self.filter_low,
            "filter_high": self.filter_high,
            "filter_notch": self.filter_notch,
            "window_sec": self.window_sec,
            "stride_sec": self.stride_sec,
            "accept_but_unused": [],  # all used in this dataset
        }

        qc = {
            "rules": {
                "reject_nonfinite": True,
                "reject_amplitude_uv": self.max_amplitude_uv,
            },
            "counts": total_qc,
        }

        ds = {
            "dataset_name": RESTING_STATE_EEG_INFO.dataset_name,
            "task_type": RESTING_STATE_EEG_INFO.task_type.value,
            "downstream_task_type": RESTING_STATE_EEG_INFO.downstream_task_type.value,
            "num_subjects_found": len(self.subj_files),
            "num_subjects_processed": len(subject_ids),
            "subjects": subject_ids,
            "channels_hint": RESTING_STATE_EEG_INFO.channels,
            "montage": RESTING_STATE_EEG_INFO.montage,
            "builder_version": BUILDER_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "segmentation": segmentation,
            "labeling": labeling,
            "preprocessing": preprocessing,
            "qc": qc,
            "unit_inference_summary": unit_summary,
            "notes": [
                "Original recording may be 2048 Hz (BioSemi) but MAT exports are commonly downsampled (e.g., 256 Hz). "
                "Builder uses per-file metadata if available; otherwise defaults to DatasetInfo.sampling_rate.",
                "If dataset duration differs across sources (e.g., 2.5 vs 5 min per condition), the builder does not assume duration; "
                "it segments whatever is present in each MAT file.",
            ],
        }
        return ds


# =========================
# CLI entrypoint (v1.4)
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Build RestingStateEEG HDF5 dataset (EO/EC).")
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, default="./hdf5_datasets", help="Output directory")

    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of subject IDs to process (e.g., S02 S03). Default: all found.",
    )

    # unified preprocessing args (defaults are group-standard)
    parser.add_argument("--target_sfreq", type=float, default=200.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--filter_low", type=float, default=0.1)
    parser.add_argument("--filter_high", type=float, default=75.0)
    parser.add_argument("--filter_notch", type=float, default=50.0)

    # QC
    parser.add_argument("--max_amplitude_uv", type=float, default=600.0)

    args = parser.parse_args()

    builder = RestingStateEEGBuilder(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_notch=args.filter_notch,
        max_amplitude_uv=args.max_amplitude_uv,
    )
    builder.build_all()


if __name__ == "__main__":
    main()
