import os
import cv2
from tqdm import tqdm

from video_loader import sample_frames  # IMPORTANT: uses src/video_loader.py

def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def filter_blurry_frames(frames, threshold=100):
    """
    frames: list of (frame_idx, frame)
    returns: list of (frame_idx, frame, blur_score)
    """
    clean_frames = []
    for idx, frame in frames:
        score = blur_score(frame)
        if score > threshold:
            clean_frames.append((idx, frame, score))
    return clean_frames

def process_video_batch(
    video_records,
    output_dir,
    sample_rate=30,
    blur_threshold=100,
    max_frames_per_video=None
):
    """
    Batch pipeline:
    - samples frames
    - filters blurry frames
    - saves cleaned frames to:
      output_dir/domain/video_stem/frame_000123.jpg

    Returns a list of logs (dicts).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for rec in tqdm(video_records, desc="Batch preprocessing"):
        domain = rec["domain"]
        video_path = rec["video_path"]
        video_stem = os.path.splitext(rec["video_name"])[0]

        save_dir = os.path.join(output_dir, domain, video_stem)
        os.makedirs(save_dir, exist_ok=True)

        try:
            frames = sample_frames(video_path, sample_rate=sample_rate, max_frames=max_frames_per_video)
            clean_frames = filter_blurry_frames(frames, threshold=blur_threshold)

            for idx, frame, score in clean_frames:
                out_path = os.path.join(save_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_path, frame)

            results.append({
                "domain": domain,
                "video_name": rec["video_name"],
                "video_path": video_path,
                "sample_rate": sample_rate,
                "total_sampled_frames": len(frames),
                "clean_frames_saved": len(clean_frames),
                "blur_threshold": blur_threshold,
                "status": "success"
            })

        except Exception as e:
            results.append({
                "domain": domain,
                "video_name": rec["video_name"],
                "video_path": video_path,
                "status": "failed",
                "error": str(e)
            })

    return results