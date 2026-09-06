import pandas as pd
from pathlib import Path
from tqdm import tqdm

from intent_inference import infer_intent_for_segment

def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()

def run_batch_intent_inference(
    kinematics_root_dir,
    objects_root_dir,
    segments_root_dir,
    out_root_dir
):
    """
    Reads:
      - outputs/kinematics/<domain>/<video>_kinematics.csv
      - outputs/object_detections/<domain>/<video>_objects.csv
      - outputs/segments/<domain>/<video>_segments.csv

    Writes:
      - outputs/intent/<domain>/<video>_intent.csv
    """
    kinematics_root_dir = Path(kinematics_root_dir)
    objects_root_dir = Path(objects_root_dir)
    segments_root_dir = Path(segments_root_dir)
    out_root_dir = Path(out_root_dir)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    kinematic_csvs = list(kinematics_root_dir.rglob("*_kinematics.csv"))
    summary = []

    for kin_csv in tqdm(sorted(kinematic_csvs), desc="Intent inference (videos)"):
        domain = kin_csv.parent.name
        video_stem = kin_csv.name.replace("_kinematics.csv", "")

        obj_csv = objects_root_dir / domain / f"{video_stem}_objects.csv"
        seg_csv = segments_root_dir / domain / f"{video_stem}_segments.csv"

        kin_df = safe_read_csv(kin_csv)
        obj_df = safe_read_csv(obj_csv)
        seg_df = safe_read_csv(seg_csv)

        if kin_df.empty or seg_df.empty:
            summary.append({
                "domain": domain,
                "video_stem": video_stem,
                "segments": 0,
                "status": "skipped_empty"
            })
            continue

        intent_rows = []

        for idx, seg in seg_df.iterrows():
            s, e = int(seg["start_frame"]), int(seg["end_frame"])

            # Get kinematic row for this segment (match by start_frame)
            kin_row = kin_df[
                (kin_df["start_frame"] == s) & (kin_df["end_frame"] == e)
            ]
            if kin_row.empty:
                continue
            kin_row = kin_row.iloc[0].to_dict()

            # Get objects in this segment's frame range
            obj_seg = obj_df[
                (obj_df["frame_id"] >= s) & (obj_df["frame_id"] <= e)
            ] if not obj_df.empty else pd.DataFrame()

            # Infer intent
            intent_info = infer_intent_for_segment(domain, obj_seg, kin_row)

            intent_rows.append({
                "domain": domain,
                "video_stem": video_stem,
                "start_frame": s,
                "end_frame": e,
                "duration_frames": int(seg.get("duration_frames", e - s + 1)),
                **intent_info
            })

        # Write output
        out_dir = out_root_dir / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_stem}_intent.csv"
        pd.DataFrame(intent_rows).to_csv(out_path, index=False)

        summary.append({
            "domain": domain,
            "video_stem": video_stem,
            "segments": len(intent_rows),
            "intent_csv": str(out_path),
            "status": "success"
        })

    return pd.DataFrame(summary)