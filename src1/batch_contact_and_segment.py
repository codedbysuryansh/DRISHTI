import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from contact_estimator import estimate_contact_for_frame
from action_segmenter import segments_from_contact_frames


def safe_read_csv(path):
    """Reads CSV safely. Returns empty DataFrame if missing/corrupt/empty."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def run_batch_contact_and_segmentation(
    hand_root_dir,
    obj_root_dir,
    out_root_dir,
    dist_thresh_px=25,
    min_len_frames=5,
    gap_merge_frames=3
):
    """
    Reads:
      - outputs/hand_landmarks/<domain>/<video>_landmarks.csv
      - outputs/object_detections/<domain>/<video>_objects.csv

    Writes:
      - outputs/contacts/<domain>/<video>_contacts.csv
      - outputs/segments/<domain>/<video>_segments.csv
    """

    hand_root_dir = Path(hand_root_dir)
    obj_root_dir = Path(obj_root_dir)
    out_root_dir = Path(out_root_dir)

    contacts_root = out_root_dir / "contacts"
    segments_root = out_root_dir / "segments"

    contacts_root.mkdir(parents=True, exist_ok=True)
    segments_root.mkdir(parents=True, exist_ok=True)

    hand_csvs = list(hand_root_dir.rglob("*_landmarks.csv"))
    summary = []

    for hand_csv in tqdm(sorted(hand_csvs), desc="Contact+Segmentation (videos)"):

        try:
            domain = hand_csv.parent.name
            video_stem = hand_csv.name.replace("_landmarks.csv", "")

            obj_csv = obj_root_dir / domain / f"{video_stem}_objects.csv"

            if not obj_csv.exists():
                summary.append({
                    "domain": domain,
                    "video_stem": video_stem,
                    "status": "skipped_no_objects"
                })
                continue

            # Safe CSV read
            hdf = safe_read_csv(hand_csv)
            odf = safe_read_csv(obj_csv)

            if hdf.empty or odf.empty:
                summary.append({
                    "domain": domain,
                    "video_stem": video_stem,
                    "frames_total": 0,
                    "contact_events": 0,
                    "segments": 0,
                    "status": "skipped_empty_csv"
                })
                continue

            # Validate frame_id column
            if "frame_id" not in hdf.columns or "frame_id" not in odf.columns:
                summary.append({
                    "domain": domain,
                    "video_stem": video_stem,
                    "status": "skipped_missing_frame_id"
                })
                continue

            # Safe numeric conversion
            hdf["frame_id"] = pd.to_numeric(hdf["frame_id"], errors="coerce")
            odf["frame_id"] = pd.to_numeric(odf["frame_id"], errors="coerce")

            hdf = hdf.dropna(subset=["frame_id"]).copy()
            odf = odf.dropna(subset=["frame_id"]).copy()

            if hdf.empty or odf.empty:
                summary.append({
                    "domain": domain,
                    "video_stem": video_stem,
                    "status": "skipped_invalid_frame_id"
                })
                continue

            hdf["frame_id"] = hdf["frame_id"].astype(int)
            odf["frame_id"] = odf["frame_id"].astype(int)

            # Build frame list
            all_frames = sorted(
                set(hdf["frame_id"].unique()) |
                set(odf["frame_id"].unique())
            )

            # Contact estimation
            contact_rows = []

            for fid in all_frames:
                h_f = hdf[hdf["frame_id"] == fid]
                o_f = odf[odf["frame_id"] == fid]

                events = estimate_contact_for_frame(
                    h_f,
                    o_f,
                    dist_thresh_px=dist_thresh_px
                )

                for ev in events:
                    ev.update({
                        "domain": domain,
                        "video_stem": video_stem,
                        "frame_id": int(fid)
                    })

                contact_rows.extend(events)

            cdf = pd.DataFrame(contact_rows)

            # Save contacts
            (contacts_root / domain).mkdir(parents=True, exist_ok=True)
            contacts_out = contacts_root / domain / f"{video_stem}_contacts.csv"
            cdf.to_csv(contacts_out, index=False)

            # Build frame-wise contact labels
            frame_contact = pd.DataFrame({"frame_id": all_frames})

            if not cdf.empty:
                contacted_frames = set(cdf["frame_id"].unique())
                frame_contact["contact"] = frame_contact["frame_id"].apply(
                    lambda x: x in contacted_frames
                )
            else:
                frame_contact["contact"] = False

            # Segment actions
            seg_df = segments_from_contact_frames(
                frame_contact,
                min_len_frames=min_len_frames,
                gap_merge_frames=gap_merge_frames
            )

            if not seg_df.empty:
                seg_df.insert(0, "video_stem", video_stem)
                seg_df.insert(0, "domain", domain)

            # Save segments
            (segments_root / domain).mkdir(parents=True, exist_ok=True)
            seg_out = segments_root / domain / f"{video_stem}_segments.csv"
            seg_df.to_csv(seg_out, index=False)

            # Summary row
            summary.append({
                "domain": domain,
                "video_stem": video_stem,
                "frames_total": len(all_frames),
                "contact_events": len(cdf),
                "segments": len(seg_df),
                "contacts_csv": str(contacts_out),
                "segments_csv": str(seg_out),
                "status": "success"
            })

        except Exception as e:
            summary.append({
                "domain": hand_csv.parent.name,
                "video_stem": hand_csv.name.replace("_landmarks.csv", ""),
                "status": f"error: {str(e)}"
            })

    return pd.DataFrame(summary)