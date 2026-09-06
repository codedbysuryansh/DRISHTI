import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from object_detector import ObjectDetectorYOLO, small_object_proposals

def run_batch_object_detection(
    frames_root_dir,
    out_root_dir,
    model_name="yolov8n.pt",
    conf=0.25,
    iou=0.5,
    save_annotated_examples=True,
    max_annotated_examples_per_video=10,
    enable_small_object_proposals=True
):
    frames_root_dir = Path(frames_root_dir)
    out_root_dir = Path(out_root_dir)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    detector = ObjectDetectorYOLO(model_name=model_name, conf=conf, iou=iou)

    annotated_dir = out_root_dir / "_annotated_examples"
    if save_annotated_examples:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # leaf video dirs (contain jpgs)
    leaf_video_dirs = [p for p in frames_root_dir.rglob("*") if p.is_dir() and list(p.glob("*.jpg"))]

    summary = []

    for vd in tqdm(sorted(leaf_video_dirs), desc="Object detection (videos)"):
        domain = vd.parent.name
        video_stem = vd.name

        rows_all = []
        annotated_saved = 0

        frame_paths = sorted(vd.glob("*.jpg"))
        for fp in tqdm(frame_paths, desc=f"{domain}/{video_stem}", leave=False):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            yres = detector.infer(frame)
            det_rows = detector.extract_detections(yres)

            frame_id = fp.stem.replace("frame_", "")

            # add frame info
            for r in det_rows:
                r.update({
                    "domain": domain,
                    "video_stem": video_stem,
                    "frame_file": fp.name,
                    "frame_id": frame_id,
                    "source": "yolo"
                })
            rows_all.extend(det_rows)

            # optional small-object proposals (useful for jewellery)
            if enable_small_object_proposals:
                props = small_object_proposals(frame)
                for (x1, y1, x2, y2) in props:
                    rows_all.append({
                        "domain": domain,
                        "video_stem": video_stem,
                        "frame_file": fp.name,
                        "frame_id": frame_id,
                        "label": "small_object_proposal",
                        "class_id": -1,
                        "confidence": 1.0,
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "source": "heuristic"
                    })

            # save annotated examples
            if save_annotated_examples and annotated_saved < max_annotated_examples_per_video:
                if det_rows:  # save only if at least one detection
                    ann = detector.draw(frame, det_rows)

                    # also draw proposals in red
                    if enable_small_object_proposals:
                        props = [r for r in rows_all if r.get("frame_file")==fp.name and r["label"]=="small_object_proposal"]
                        for r in props[:20]:
                            x1, y1, x2, y2 = map(int, [r["x1"], r["y1"], r["x2"], r["y2"]])
                            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 255), 1)

                    save_path = annotated_dir / domain / video_stem
                    save_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path / fp.name), ann)
                    annotated_saved += 1

        # write per-video CSV
        (out_root_dir / domain).mkdir(parents=True, exist_ok=True)
        out_file = out_root_dir / domain / f"{video_stem}_objects.csv"
        pd.DataFrame(rows_all).to_csv(out_file, index=False)

        summary.append({
            "domain": domain,
            "video_stem": video_stem,
            "frames_in_folder": len(frame_paths),
            "detection_rows_total": len(rows_all),
            "csv_path": str(out_file),
            "annotated_examples_saved": annotated_saved
        })

    return pd.DataFrame(summary)