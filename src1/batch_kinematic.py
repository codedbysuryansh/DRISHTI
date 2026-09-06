import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from kinematic_analyzer import (
    per_frame_hand_kinematics,
    aggregate_segment_kinematics
)

def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()

def run_batch_kinematics(
    hand_root_dir,
    segments_root_dir,
    out_root_dir
):
    """
    Reads:
      - outputs/hand_landmarks/<domain>/<video>_landmarks.csv
      - outputs/segments/<domain>/<video>_segments.csv

    Writes:
      - outputs/kinematics/<domain>/<video>_kinematics.csv
        (one row per segment with kinematic summary)
    """
    hand_root_dir = Path(hand_root_dir)
    segments_root_dir = Path(segments_root_dir)
    out_root_dir = Path(out_root_dir)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    seg_csvs = list(segments_root_dir.rglob("*_segments.csv"))
    summary = []

    for seg_csv in tqdm(sorted(seg_csvs), desc="Kinematic analysis (videos)"):
        domain = seg_csv.parent.name
        video_stem = seg_csv.name.replace("_segments.csv", "")

        hand_csv = hand_root_dir / domain / f"{video_stem}_landmarks.csv"

        seg_df = safe_read_csv(seg_csv)
        hand_df = safe_read_csv(hand_csv)

        if seg_df.empty or hand_df.empty:
            summary.append({
                "domain": domain,
                "video_stem": video_stem,
                "segments": 0,
                "status": "skipped_empty"
            })
            continue

        if "frame_id" not in hand_df.columns:
            summary.append({
                "domain": domain, "video_stem": video_stem,
                "status": "skipped_no_frame_id"
            })
            continue

        hand_df["frame_id"] = pd.to_numeric(hand_df["frame_id"], errors="coerce").dropna().astype(int)

        seg_rows = []

        # Pre-compute per-frame, per-hand kinematics once
        per_frame_records = []
        # group by (frame_id, hand_id)
        grouped = hand_df.groupby(["frame_id", "hand_id"])
        for (fid, hid), gdf in grouped:
            feats = per_frame_hand_kinematics(gdf)
            if feats is None:
                continue
            handed = None
            if "handedness" in gdf.columns and gdf["handedness"].notna().any():
                handed = gdf["handedness"].dropna().iloc[0]
            feats.update({
                "frame_id": int(fid),
                "hand_id": int(hid),
                "handedness": handed
            })
            per_frame_records.append(feats)

        per_frame_df = pd.DataFrame(per_frame_records)

        for _, seg in seg_df.iterrows():
            s, e = int(seg["start_frame"]), int(seg["end_frame"])
            seg_frames = per_frame_df[(per_frame_df["frame_id"] >= s) & (per_frame_df["frame_id"] <= e)]

            agg = aggregate_segment_kinematics(seg_frames)

            seg_rows.append({
                "domain": domain,
                "video_stem": video_stem,
                "start_frame": s,
                "end_frame": e,
                "duration_frames": int(seg.get("duration_frames", e - s + 1)),
                "dominant_grasp": agg["dominant_grasp"],
                "grasp_distribution": json.dumps(agg["grasp_distribution"]),
                "motion_speed_px": agg["motion_speed_px"],
                "motion_label": agg["motion_label"],
                "hands_used": agg["hands_used"],
                "dominant_handedness": agg["dominant_handedness"],
                "bimanual": agg["bimanual"]
            })

        out_dir = out_root_dir / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_stem}_kinematics.csv"
        pd.DataFrame(seg_rows).to_csv(out_path, index=False)

        summary.append({
            "domain": domain,
            "video_stem": video_stem,
            "segments": len(seg_rows),
            "kinematics_csv": str(out_path),
            "status": "success"
        })

    return pd.DataFrame(summary)