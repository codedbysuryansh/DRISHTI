import cv2
from pathlib import Path

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

def scan_videos(data_dir, domains):
    """
    Backward-compatible scanner (your notebook 01 likely uses this).
    """
    records = []
    data_dir = Path(data_dir)

    for domain in domains:
        domain_path = data_dir / domain
        if not domain_path.exists():
            continue

        for file in domain_path.iterdir():
            if file.suffix.lower() in VIDEO_EXTENSIONS:
                records.append({
                    "domain": domain,
                    "video_name": file.name,
                    "video_path": str(file)
                })
    return records

def get_all_video_paths(data_dir, domains):
    """
    New name (batch mode). Returns same structure as scan_videos.
    """
    return scan_videos(data_dir, domains)

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration
    }

def sample_frames(video_path, sample_rate=30, max_frames=None):
    """
    Extract frames at a stride of sample_rate.
    max_frames: optional cap for debugging.
    Returns: list of (frame_idx, frame)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    sampled = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frames.append((frame_idx, frame))
            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames