"""EntitiesEEG - Audio modality (40-class concept) builder (final)

Template trigger extraction:
- Map raw.annotations description codes via `code_to_name`
- Use `mne.events_from_annotations` to get (events, event_id)
- Blocks: "Audio_1_trial_start" ... "Audio_8_trial_end"

Segmentation:
- duration_log_audio_group<block>.csv  (block = 0..7)
- Expand ONLY stimulus rows of current modality; ISI/gray rows excluded
- Each segment window length == stimulus Duration(s) from duration_log

Preprocessing:
- EEG-only
- notch 50 Hz
- bandpass 0.1–75 Hz
- resample to 200 Hz
- store in microvolts (uV)

QC:
- reject if any non-finite
- reject if max(|x|) > 600 uV

Quick debug (1 subject, 1 block, compare with legacy labels):
  python entities_audio40_final.py "Z:\Datasets\Entities" --output_dir "Z:\Processed_datasets\EEG_Bench" --subjects sub3 --debug_block 0 --compare_labels "Z:\Datasets\Entities\Datasets\audio_labels.npy"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json as _json

import numpy as np
import pandas as pd
import mne

# ---- local imports (support both package and standalone usage) ----
try:
    from ..schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from ..hdf5_io import HDF5Writer
    from ..config import DatasetInfo, DatasetTaskType, DownstreamTaskType, PreprocConfig
except Exception:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import SubjectAttrs, TrialAttrs, SegmentAttrs
    from hdf5_io import HDF5Writer
    from config import DatasetInfo, DatasetTaskType, DownstreamTaskType, PreprocConfig


DATASET_INFO = DatasetInfo(
    dataset_name="EntitiesAudio40",
    task_type=DatasetTaskType.COGNITIVE,
    downstream_task_type=DownstreamTaskType.CLASSIFICATION,
    num_labels=40,
    category_list=[f"{i:03d}" for i in range(1, 41)],
    sampling_rate=0.0,
    montage="standard_1020",
    channels=[],
)

END_TRUST_MARGIN_SEC = 0.10  # seconds

code_to_name = {
    "11": "Image_1_enter",
    "12": "Image_2_enter",
    "13": "Image_3_enter",
    "14": "Image_4_enter",
    "15": "Image_5_enter",
    "16": "Image_6_enter",
    "17": "Image_7_enter",
    "18": "Image_8_enter",
    "21": "Image_1_trial_start",
    "22": "Image_2_trial_start",
    "23": "Image_3_trial_start",
    "24": "Image_4_trial_start",
    "25": "Image_5_trial_start",
    "26": "Image_6_trial_start",
    "27": "Image_7_trial_start",
    "28": "Image_8_trial_start",
    "31": "Image_1_trial_end",
    "32": "Image_2_trial_end",
    "33": "Image_3_trial_end",
    "34": "Image_4_trial_end",
    "35": "Image_5_trial_end",
    "36": "Image_6_trial_end",
    "37": "Image_7_trial_end",
    "38": "Image_8_trial_end",
    "111": "Text_1_enter",
    "112": "Text_2_enter",
    "113": "Text_3_enter",
    "114": "Text_4_enter",
    "115": "Text_5_enter",
    "116": "Text_6_enter",
    "117": "Text_7_enter",
    "118": "Text_8_enter",
    "121": "Text_1_trial_start",
    "122": "Text_2_trial_start",
    "123": "Text_3_trial_start",
    "124": "Text_4_trial_start",
    "125": "Text_5_trial_start",
    "126": "Text_6_trial_start",
    "127": "Text_7_trial_start",
    "128": "Text_8_trial_start",
    "131": "Text_1_trial_end",
    "132": "Text_2_trial_end",
    "133": "Text_3_trial_end",
    "134": "Text_4_trial_end",
    "135": "Text_5_trial_end",
    "136": "Text_6_trial_end",
    "137": "Text_7_trial_end",
    "138": "Text_8_trial_end",
    "211": "Audio_1_enter",
    "212": "Audio_2_enter",
    "213": "Audio_3_enter",
    "214": "Audio_4_enter",
    "215": "Audio_5_enter",
    "216": "Audio_6_enter",
    "217": "Audio_7_enter",
    "218": "Audio_8_enter",
    "221": "Audio_1_trial_start",
    "222": "Audio_2_trial_start",
    "223": "Audio_3_trial_start",
    "224": "Audio_4_trial_start",
    "225": "Audio_5_trial_start",
    "226": "Audio_6_trial_start",
    "227": "Audio_7_trial_start",
    "228": "Audio_8_trial_start",
    "231": "Audio_1_trial_end",
    "232": "Audio_2_trial_end",
    "233": "Audio_3_trial_end",
    "234": "Audio_4_trial_end",
    "235": "Audio_5_trial_end",
    "236": "Audio_6_trial_end",
    "237": "Audio_7_trial_end",
    "238": "Audio_8_trial_end",
}


def _normalize_chn(name: str) -> str:
    return name.strip().upper().replace(" ", "")


BAD_CH_PREFIXES = ("ECG", "EOG", "HEO", "VEO")

def _is_bad_channel(ch: str) -> bool:
    c = _normalize_chn(ch)
    return c.startswith(BAD_CH_PREFIXES)


def _find_bdf_for_subject(subject_dir: Path) -> Optional[Path]:
    cands = list(subject_dir.rglob("1.bdf"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        for p in cands:
            if "离线数据" in str(p):
                return p
        return cands[0]
    return None


def _apply_code_to_name(raw: mne.io.BaseRaw) -> None:
    if raw.annotations is None or len(raw.annotations) == 0:
        return
    new_desc: List[str] = []
    for desc in raw.annotations.description:
        d = str(desc)
        new_desc.append(code_to_name.get(d, d))
    raw.set_annotations(
        mne.Annotations(
            onset=raw.annotations.onset,
            duration=raw.annotations.duration,
            description=new_desc,
        )
    )


def _load_events_with_template(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    _apply_code_to_name(raw)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id


def _read_duration_log(duration_log_dir: Path, modality: str, group_idx: int) -> pd.DataFrame:
    p = duration_log_dir / f"duration_log_{modality}_group{group_idx}.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def _sum_duration(df: pd.DataFrame) -> float:
    return float(df["Duration(s)"].astype(float).sum())


def _expand_stimulus_events(df: pd.DataFrame, modality: str) -> List[Tuple[float, float, int]]:
    t = 0.0
    out: List[Tuple[float, float, int]] = []
    stim_type = modality.upper()
    for _, row in df.iterrows():
        typ = str(row["Type"]).strip().upper()
        dur = float(row["Duration(s)"])
        if typ == "ISI":
            t += dur
            continue
        if typ != stim_type:
            t += dur
            continue
        concept = str(row["Concept"]).strip()
        try:
            cid = int(concept)  # 1..40
        except Exception:
            t += dur
            continue
        out.append((t, dur, cid))
        t += dur
    return out


def _find_first_after(events: np.ndarray, code: int, after_sample: int) -> Optional[int]:
    for e in events:
        if int(e[0]) <= after_sample:
            continue
        if int(e[2]) == int(code):
            return int(e[0])
    return None


def _extract_blocks(events: np.ndarray, event_id: Dict[str, int], sfreq0: float, modality: str) -> List[Dict]:
    blocks: List[Dict] = []
    cursor = -1
    mod = modality.capitalize()
    for i in range(1, 9):
        start_key = f"{mod}_{i}_trial_start"
        end_key = f"{mod}_{i}_trial_end"
        if start_key not in event_id:
            blocks.append({"ok": False, "block_idx": i - 1, "reason": f"missing_event_id:{start_key}"})
            continue
        start_code = int(event_id[start_key])
        end_code = int(event_id[end_key]) if end_key in event_id else None

        s_samp = _find_first_after(events, start_code, cursor)
        if s_samp is None:
            blocks.append({"ok": False, "block_idx": i - 1, "reason": f"missing_start_event:{start_key}"})
            continue

        e_samp = None
        if end_code is not None:
            e_samp = _find_first_after(events, end_code, s_samp)

        blocks.append(
            {
                "ok": True,
                "block_idx": i - 1,
                "start_key": start_key,
                "end_key": end_key,
                "start_sec": s_samp / sfreq0,
                "end_sec": (e_samp / sfreq0) if e_samp is not None else None,
            }
        )
        cursor = e_samp if e_samp is not None else s_samp
    return blocks


def _write_dataset_info(out_dir: Path, channels: List[str], original_sfreq: float, preproc: PreprocConfig, stats: Dict, note: str):
    payload = {
        "dataset": {
            "name": "EntitiesAudio40",
            "task_type": "cognitive",
            "downstream_task": "classification",
            "num_labels": 40,
            "category_list": [f"{i:03d}" for i in range(1, 41)],
            "original_sampling_rate": float(original_sfreq),
            "channels": channels,
            "montage": "standard_1020",
            "note": note,
        },
        "processing": {
            "target_sampling_rate": float(preproc.target_sfreq),
            "filter_low": float(preproc.filter_low),
            "filter_high": float(preproc.filter_high),
            "filter_notch": float(preproc.filter_notch),
            "unit": "uV",
            "baseline_correction": False,
            "max_amplitude_uv": 600.0,
            "segmentation": "stimulus_duration_from_duration_log",
        },
        "statistics": stats,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_info.json").write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class EntitiesAudio40BuilderFinal:
    def __init__(
        self,
        raw_root: str,
        output_dir: str,
        duration_log_dir: Optional[str] = None,
        target_sfreq: float = 200.0,
        filter_low: float = 0.1,
        filter_high: float = 75.0,
        filter_notch: float = 50.0,
        max_amplitude_uv: float = 600.0,
    ):
        self.raw_root = Path(raw_root)
        self.rawdata_dir = self.raw_root
        self.duration_log_dir = Path(duration_log_dir) if duration_log_dir else (self.raw_root / "duration_log")
        self.out_dir = Path(output_dir) / "EntitiesAudio40"

        self.preproc = PreprocConfig(
            filter_low=filter_low,
            filter_high=filter_high,
            filter_notch=filter_notch,
            target_sfreq=target_sfreq,
            window_sec=1.0,
            stride_sec=1.0,
            output_dir=str(output_dir),
        )
        self.max_amplitude_uv = float(max_amplitude_uv)
        self.channels_final: List[str] = []

        self._stats: Dict = {
            "total_subjects": 0,
            "successful": 0,
            "failed": 0,
            "failed_subject_ids": [],
            "subjects": {},
            "segments_expected_per_subject": 1600,
            "segments_written_total": 0,
            "segments_rejected_total": 0,
            "reject_breakdown": {"rej_boundary": 0, "rej_nonfinite": 0, "rej_amp": 0},
            "end_trigger_overridden_by_duration_log": 0,
        }

    def get_subject_dirs(self, subjects: Optional[List[str]] = None) -> List[Path]:
        if not self.rawdata_dir.exists():
            raise FileNotFoundError(f"raw_root not found: {self.rawdata_dir}")
        subs = sorted([p for p in self.rawdata_dir.glob("sub*") if p.is_dir()])
        subs = [p for p in subs if _find_bdf_for_subject(p) is not None]
        if subjects:
            want = set(subjects)
            subs = [p for p in subs if p.name in want]
        return subs

    def scan_channels_intersection(self, subjects: Optional[List[str]] = None, max_subjects: Optional[int] = None) -> List[str]:
        eeg_sets = []
        subs = self.get_subject_dirs(subjects=subjects)
        if max_subjects is not None:
            subs = subs[:max_subjects]
        for sub_dir in subs:
            bdf = _find_bdf_for_subject(sub_dir)
            if bdf is None:
                continue
            raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
            chs = [_normalize_chn(raw.ch_names[i]) for i in picks]
            chs = [c for c in chs if not _is_bad_channel(c)]
            eeg_sets.append(set(chs))
            raw.close()
        if not eeg_sets:
            raise RuntimeError("Could not scan any subject for EEG channels.")
        inter = set.intersection(*eeg_sets)
        self.channels_final = sorted([c for c in inter if not _is_bad_channel(c)])
        return self.channels_final

    def _prep_block(self, raw: mne.io.BaseRaw, tmin: float, tmax: float) -> np.ndarray:
        blk = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        blk.load_data()

        picks = mne.pick_types(blk.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
        blk.pick(picks)
        blk.rename_channels({ch: _normalize_chn(ch) for ch in blk.ch_names})
        # Drop channels that are not EEG but may be mis-tagged as EEG in BDF
        bad_now = [c for c in blk.ch_names if _is_bad_channel(c)]
        if bad_now:
            blk.drop_channels(bad_now)
        # Drop non-EEG channels that may be mislabeled as EEG in BDF (common in BioSemi exports)
        drop_prefix = ("HEO", "VEO", "EOG", "ECG")
        drop_names = [ch for ch in blk.ch_names if ch.startswith(drop_prefix)]
        if drop_names:
            blk.drop_channels(drop_names)

        if not self.channels_final:
            raise RuntimeError("channels_final empty. Call scan_channels_intersection first.")
        missing = [c for c in self.channels_final if c not in blk.ch_names]
        if missing:
            raise ValueError(f"missing channels: {missing}")
        blk.pick_channels(self.channels_final, ordered=True)

        if self.preproc.filter_notch and self.preproc.filter_notch > 0:
            blk.notch_filter(freqs=[self.preproc.filter_notch], verbose=False)
        blk.filter(l_freq=self.preproc.filter_low, h_freq=self.preproc.filter_high, verbose=False)

        if abs(blk.info["sfreq"] - self.preproc.target_sfreq) > 1e-6:
            blk.resample(self.preproc.target_sfreq, verbose=False)

        return (blk.get_data() * 1e6).astype(np.float32)  # V->uV

    def build_subject(self, sub_dir: Path, debug_block: Optional[int] = None, compare_labels: Optional[str] = None) -> Optional[str]:
        sub_id = sub_dir.name
        self._stats["total_subjects"] += 1
        self._stats["subjects"][sub_id] = {"blocks_ok": 0, "segments_written": 0, "segments_rejected": 0}

        bdf = _find_bdf_for_subject(sub_dir)
        if bdf is None:
            self._stats["failed"] += 1
            self._stats["failed_subject_ids"].append(sub_id)
            return None

        raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose=False)
        sfreq0 = float(raw.info["sfreq"])
        events, event_id = _load_events_with_template(raw)

        if not self.channels_final:
            raise RuntimeError("channels_final empty (call scan_channels_intersection).")

        blocks = _extract_blocks(events, event_id, sfreq0, "Audio")

        legacy = np.load(compare_labels).astype(int).ravel() if compare_labels else None
        legacy_cursor = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"sub_{sub_id}.h5"

        subject_attrs = SubjectAttrs(
            subject_id=sub_id,
            dataset_name="EntitiesAudio40",
            task_type=DATASET_INFO.task_type.value,
            downstream_task_type=DownstreamTaskType.CLASSIFICATION.value,
            rsFreq=float(self.preproc.target_sfreq),
            chn_name=self.channels_final,
            num_labels=40,
            category_list=[f"{i:03d}" for i in range(1, 41)],
            chn_pos=None,
            chn_ori=None,
            chn_type="EEG",
            montage=DATASET_INFO.montage,
        )

        seg_global_id = 0
        with HDF5Writer(str(out_path), subject_attrs) as writer:
            for blk in blocks:
                if not blk.get("ok", False):
                    continue
                bi = int(blk["block_idx"])
                if debug_block is not None and bi != int(debug_block):
                    continue

                df = _read_duration_log(self.duration_log_dir, "audio", bi)
                stim_events = _expand_stimulus_events(df, "audio")
                total_dur = _sum_duration(df)

                start_sec = float(blk["start_sec"])
                end_sec_trigger = blk.get("end_sec", None)
                end_sec_log = start_sec + total_dur

                end_sec = end_sec_log if end_sec_trigger is None else float(end_sec_trigger)
                if end_sec_trigger is not None and end_sec < end_sec_log - END_TRUST_MARGIN_SEC:
                    end_sec = end_sec_log
                    self._stats["end_trigger_overridden_by_duration_log"] += 1

                trial_name = writer.add_trial(TrialAttrs(trial_id=bi, session_id=0, task_name="audio"))
                Xblk = self._prep_block(raw, tmin=start_sec, tmax=end_sec)
                sf = float(self.preproc.target_sfreq)

                for onset_in_blk, dur_sec, label_1to40 in stim_events:
                    s0 = int(round(float(onset_in_blk) * sf))
                    s1 = int(round((float(onset_in_blk) + float(dur_sec)) * sf))
                    # Compare against legacy labels per-STIMULUS index (advance cursor regardless of QC keep/reject)
                    if legacy is not None and legacy_cursor < legacy.shape[0]:
                        if int(legacy[legacy_cursor]) != int(label_1to40):
                            print(f"[LABEL_MISMATCH] {sub_id} block{bi} stim{legacy_cursor} legacy={int(legacy[legacy_cursor])} new={int(label_1to40)}")
                        legacy_cursor += 1
                    if s0 < 0 or s1 > Xblk.shape[1] or s1 <= s0:
                        self._stats["reject_breakdown"]["rej_boundary"] += 1
                        self._stats["segments_rejected_total"] += 1
                        self._stats["subjects"][sub_id]["segments_rejected"] += 1
                        continue

                    seg = Xblk[:, s0:s1]
                    if not np.isfinite(seg).all():
                        self._stats["reject_breakdown"]["rej_nonfinite"] += 1
                        self._stats["segments_rejected_total"] += 1
                        self._stats["subjects"][sub_id]["segments_rejected"] += 1
                        continue
                    if float(np.max(np.abs(seg))) > self.max_amplitude_uv:
                        self._stats["reject_breakdown"]["rej_amp"] += 1
                        self._stats["segments_rejected_total"] += 1
                        self._stats["subjects"][sub_id]["segments_rejected"] += 1
                        continue

                    seg_attrs = SegmentAttrs(
                        segment_id=seg_global_id,
                        start_time=float(onset_in_blk),
                        end_time=float(onset_in_blk + float(dur_sec)),
                        time_length=float(dur_sec),
                        label=np.array([int(label_1to40)], dtype=np.int64),
                        task_label=str(int(label_1to40)),
                    )
                    writer.add_segment(trial_name, seg_attrs, seg)
                    seg_global_id += 1
                    self._stats["subjects"][sub_id]["segments_written"] += 1
                    self._stats["segments_written_total"] += 1

                self._stats["subjects"][sub_id]["blocks_ok"] += 1

        raw.close()
        self._stats["successful"] += 1
        return str(out_path)

    def build_all(
        self,
        subjects: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        debug_block: Optional[int] = None,
        compare_labels: Optional[str] = None,
    ) -> List[str]:
        if not self.channels_final:
            self.scan_channels_intersection(subjects=subjects, max_subjects=max_subjects)

        subs = self.get_subject_dirs(subjects=subjects)
        if max_subjects is not None:
            subs = subs[:max_subjects]

        outs: List[str] = []
        for sub_dir in subs:
            try:
                out = self.build_subject(sub_dir, debug_block=debug_block, compare_labels=compare_labels)
                if out:
                    outs.append(out)
            except Exception as e:
                sub_id = sub_dir.name
                self._stats["failed"] += 1
                self._stats["failed_subject_ids"].append(sub_id)
                print(f"[SKIP] {sub_id}: {e}")

        note = (
            "Audio modality only; triggers via template code_to_name + events_from_annotations; "
            "stimulus windows follow duration_log (exclude ISI); store uV; notch 50."
        )
        orig_sfreq = 0.0
        if subs:
            bdf = _find_bdf_for_subject(subs[0])
            if bdf:
                r = mne.io.read_raw_bdf(str(bdf), preload=False, verbose=False)
                orig_sfreq = float(r.info["sfreq"])
                r.close()

        _write_dataset_info(self.out_dir, self.channels_final, orig_sfreq, self.preproc, self._stats, note)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=str)
    ap.add_argument("--output_dir", type=str, default="./hdf5")
    ap.add_argument("--duration_log_dir", type=str, default=None)
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--debug_block", type=int, default=None)
    ap.add_argument("--compare_labels", type=str, default=None)
    args = ap.parse_args()

    builder = EntitiesAudio40BuilderFinal(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        duration_log_dir=args.duration_log_dir,
        target_sfreq=200.0,
        filter_low=0.1,
        filter_high=75.0,
        filter_notch=50.0,
        max_amplitude_uv=600.0,
    )
    outs = builder.build_all(
        subjects=args.subjects,
        max_subjects=args.max_subjects,
        debug_block=args.debug_block,
        compare_labels=args.compare_labels,
    )
    print(f"[EntitiesAudio40] Done. H5 files: {len(outs)} -> {builder.out_dir}")


if __name__ == "__main__":
    main()
