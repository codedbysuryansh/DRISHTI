import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from hand_detector import HandDetector

def run_batch_hand_detection(
    frames_root_dir,
    out_root_dir,
    save_annotated_examples=True,
    max_annotated_examples_per_video=10
):
    """
    frames_root_dir: ../outputs/sampled_frames
      expected structure: domain/video_stem/frame_000123.jpg

    out_root_dir: ../outputs/hand_landmarks
      outputs:
        - domain/video_stem_landmarks.csv
        - optional annotated frames in ../outputs/hand_landmarks/_annotated_examples/...

    Returns: summary dataframe
    """
    frames_root_dir = Path(frames_root_dir)
    out_root_dir = Path(out_root_dir)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(static_image_mode=True)

    annotated_dir = out_root_dir / "_annotated_examples"
    if save_annotated_examples:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # find all domain/video folders
    video_dirs = [p for p in frames_root_dir.rglob("*") if p.is_dir()]
    # filter to leaf directories containing jpgs
    leaf_video_dirs = []
    for vd in video_dirs:
        if list(vd.glob("*.jpg")):
            leaf_video_dirs.append(vd)

    summary = []

    for vd in tqdm(sorted(leaf_video_dirs), desc="Hand detection (videos)"):
        # vd = .../sampled_frames/domain/video_stem
        domain = vd.parent.name
        video_stem = vd.name

        rows_all = []
        annotated_saved = 0

        frame_paths = sorted(vd.glob("*.jpg"))
        for fp in tqdm(frame_paths, desc=f"{domain}/{video_stem}", leave=False):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            results = detector.infer(frame)
            lm_rows = detector.extract_landmarks(frame, results)

            # add frame info + store
            frame_id = fp.stem.replace("frame_", "")  # "000123"
            for r in lm_rows:
                r.update({
                    "domain": domain,
                    "video_stem": video_stem,
                    "frame_file": fp.name,
                    "frame_id": frame_id
                })
            rows_all.extend(lm_rows)

            # save a few annotated examples per video for demo
            if save_annotated_examples and annotated_saved < max_annotated_examples_per_video:
                if results.multi_hand_landmarks:  # save only if hand detected
                    ann = detector.draw(frame, results)
                    save_path = annotated_dir / domain / video_stem
                    save_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path / fp.name), ann)
                    annotated_saved += 1

        # write per-video CSV
        out_csv = out_root_dir / domain
        out_csv.mkdir(parents=True, exist_ok=True)
        out_file = out_csv / f"{video_stem}_landmarks.csv"

        df = pd.DataFrame(rows_all)
        df.to_csv(out_file, index=False)

        summary.append({
            "domain": domain,
            "video_stem": video_stem,
            "frames_in_folder": len(frame_paths),
            "landmark_rows": len(rows_all),
            "csv_path": str(out_file),
            "annotated_examples_saved": annotated_saved
        })

    return pd.DataFrame(summary)