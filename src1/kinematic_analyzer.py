import numpy as np
import pandas as pd

# MediaPipe landmark IDs
LM_WRIST = 0
LM_THUMB_TIP, LM_THUMB_IP, LM_THUMB_MCP = 4, 3, 2
LM_INDEX_TIP, LM_INDEX_PIP, LM_INDEX_MCP = 8, 6, 5
LM_MIDDLE_TIP, LM_MIDDLE_PIP, LM_MIDDLE_MCP = 12, 10, 9
LM_RING_TIP, LM_RING_PIP, LM_RING_MCP = 16, 14, 13
LM_PINKY_TIP, LM_PINKY_PIP, LM_PINKY_MCP = 20, 18, 17

FINGERS = {
    "index":  (LM_INDEX_TIP, LM_INDEX_PIP, LM_INDEX_MCP),
    "middle": (LM_MIDDLE_TIP, LM_MIDDLE_PIP, LM_MIDDLE_MCP),
    "ring":   (LM_RING_TIP, LM_RING_PIP, LM_RING_MCP),
    "pinky":  (LM_PINKY_TIP, LM_PINKY_PIP, LM_PINKY_MCP),
}

def _build_pts(hand_df):
    """hand_df: rows for ONE frame & ONE hand_id. Returns dict lm_id -> (x_px,y_px)."""
    pts = {}
    for _, r in hand_df.iterrows():
        pts[int(r["landmark_id"])] = (float(r["x_px"]), float(r["y_px"]))
    return pts

def _dist(p, q):
    return float(np.hypot(p[0]-q[0], p[1]-q[1]))

def _hand_scale(pts):
    """Reference scale = wrist to middle-finger MCP distance (palm size proxy)."""
    if LM_WRIST not in pts or LM_MIDDLE_MCP not in pts:
        return None
    return _dist(pts[LM_WRIST], pts[LM_MIDDLE_MCP]) + 1e-6

def _finger_curl_ratio(pts, scale):
    """
    For each finger: ratio of (tip->mcp distance) / scale.
    Smaller ratio = more curled. Larger = more extended.
    Returns dict.
    """
    out = {}
    for name, (tip, _pip, mcp) in FINGERS.items():
        if tip in pts and mcp in pts:
            out[name] = _dist(pts[tip], pts[mcp]) / scale
        else:
            out[name] = None
    return out

def _pinch_distance(pts, scale):
    """Thumb tip ↔ index tip distance normalized by scale."""
    if LM_THUMB_TIP in pts and LM_INDEX_TIP in pts:
        return _dist(pts[LM_THUMB_TIP], pts[LM_INDEX_TIP]) / scale
    return None

def classify_grasp(curl_ratios, pinch_dist):
    """
    Heuristics tuned for normalized values:
    - pinch_dist small AND curls medium → pinch
    - all curls small → closed/power grip
    - all curls large → open hand
    - mixed → partial grip
    """
    if pinch_dist is not None and pinch_dist < 0.35:
        return "pinch"

    avg_curl = np.nanmean([v for v in curl_ratios.values() if v is not None]) if curl_ratios else None
    if avg_curl is None:
        return "unknown"

    if avg_curl < 0.55:
        return "closed_power"
    elif avg_curl > 0.95:
        return "open_hand"
    else:
        return "partial_grip"

def per_frame_hand_kinematics(hand_landmarks_df_one_frame_one_hand):
    """
    Returns dict of features for one (frame, hand).
    """
    pts = _build_pts(hand_landmarks_df_one_frame_one_hand)
    scale = _hand_scale(pts)
    if scale is None:
        return None

    curls = _finger_curl_ratio(pts, scale)
    pinch = _pinch_distance(pts, scale)
    grasp = classify_grasp(curls, pinch)

    wrist = pts.get(LM_WRIST, None)

    return {
        "wrist_x": wrist[0] if wrist else None,
        "wrist_y": wrist[1] if wrist else None,
        "scale_palm": scale,
        "pinch_dist_norm": pinch,
        "curl_index": curls.get("index"),
        "curl_middle": curls.get("middle"),
        "curl_ring": curls.get("ring"),
        "curl_pinky": curls.get("pinky"),
        "grasp": grasp
    }

def motion_intensity_from_wrist(wrist_track):
    """
    wrist_track: list of (x,y) over frames in a segment.
    Returns mean speed (px/frame).
    """
    if len(wrist_track) < 2:
        return 0.0
    speeds = []
    for i in range(1, len(wrist_track)):
        a, b = wrist_track[i-1], wrist_track[i]
        if a is None or b is None:
            continue
        speeds.append(np.hypot(b[0]-a[0], b[1]-a[1]))
    if not speeds:
        return 0.0
    return float(np.mean(speeds))

def label_motion_intensity(speed_px):
    if speed_px < 1.5:
        return "still"
    elif speed_px < 6:
        return "slow"
    elif speed_px < 18:
        return "moderate"
    else:
        return "fast"

def aggregate_segment_kinematics(seg_frames_df):
    """
    seg_frames_df: per-frame per-hand kinematic features for ONE segment.
    Columns required: hand_id, grasp, wrist_x, wrist_y, frame_id, handedness
    Returns dict summary for the segment.
    """
    if seg_frames_df.empty:
        return {
            "dominant_grasp": "unknown",
            "grasp_distribution": {},
            "motion_speed_px": 0.0,
            "motion_label": "still",
            "hands_used": 0,
            "dominant_handedness": None,
            "bimanual": False
        }

    # Dominant grasp = most frequent
    grasp_counts = seg_frames_df["grasp"].value_counts(dropna=True).to_dict()
    dominant_grasp = max(grasp_counts, key=grasp_counts.get) if grasp_counts else "unknown"

    # Hands used
    hand_ids = seg_frames_df["hand_id"].dropna().unique().tolist()
    bimanual = len(hand_ids) >= 2

    # Dominant handedness (if available)
    dom_handed = None
    if "handedness" in seg_frames_df.columns:
        hh = seg_frames_df["handedness"].dropna()
        if not hh.empty:
            dom_handed = hh.value_counts().idxmax()

    # Motion: use the most-active hand wrist track
    motion_speed = 0.0
    for hid in hand_ids:
        track = (
            seg_frames_df[seg_frames_df["hand_id"] == hid]
            .sort_values("frame_id")[["wrist_x", "wrist_y"]]
            .apply(lambda r: (r["wrist_x"], r["wrist_y"]) if pd.notna(r["wrist_x"]) else None, axis=1)
            .tolist()
        )
        s = motion_intensity_from_wrist(track)
        motion_speed = max(motion_speed, s)

    return {
        "dominant_grasp": dominant_grasp,
        "grasp_distribution": grasp_counts,
        "motion_speed_px": round(motion_speed, 3),
        "motion_label": label_motion_intensity(motion_speed),
        "hands_used": int(len(hand_ids)),
        "dominant_handedness": dom_handed,
        "bimanual": bool(bimanual)
    }