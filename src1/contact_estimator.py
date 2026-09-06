import numpy as np

# MediaPipe landmark IDs (hand)
LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_TIP = 8
LM_MIDDLE_TIP = 12
LM_RING_TIP = 16
LM_PINKY_TIP = 20

FINGERTIPS = [LM_THUMB_TIP, LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP]

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

def point_box_distance(x, y, box):
    """
    Distance from point to axis-aligned rectangle (0 if inside).
    """
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return (dx**2 + dy**2) ** 0.5

def build_hand_points(hand_df):
    """
    hand_df: rows for ONE frame & ONE hand_id
    expects x_px, y_px columns
    returns dict lm_id -> (x_px, y_px)
    """
    pts = {}
    for _, r in hand_df.iterrows():
        pts[int(r["landmark_id"])] = (float(r["x_px"]), float(r["y_px"]))
    return pts

def estimate_contact_for_frame(hand_landmarks_df, objects_df, dist_thresh_px=25):
    """
    Inputs are dataframes filtered for ONE frame:
    - hand_landmarks_df: all hands landmarks rows (can include multiple hands)
    - objects_df: all object bbox rows (YOLO + heuristic)
    Returns list of contact events rows.
    """
    contacts = []

    if hand_landmarks_df.empty or objects_df.empty:
        return contacts

    for hand_id in sorted(hand_landmarks_df["hand_id"].unique()):
        hdf = hand_landmarks_df[hand_landmarks_df["hand_id"] == hand_id]
        pts = build_hand_points(hdf)

        fingertip_pts = [pts[lm] for lm in FINGERTIPS if lm in pts]
        if not fingertip_pts:
            continue

        for _, obj in objects_df.iterrows():
            box = (float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"]))

            inside = 0
            min_dist = 1e9
            for (x, y) in fingertip_pts:
                if point_in_box(x, y, box):
                    inside += 1
                    d = 0.0
                else:
                    d = point_box_distance(x, y, box)
                min_dist = min(min_dist, d)

            contact = (inside > 0) or (min_dist <= dist_thresh_px)

            if contact:
                contacts.append({
                    "hand_id": int(hand_id),
                    "handedness": hdf["handedness"].dropna().iloc[0] if "handedness" in hdf.columns and hdf["handedness"].notna().any() else None,
                    "object_label": obj.get("label", None),
                    "object_source": obj.get("source", None),
                    "object_conf": float(obj.get("confidence", 0.0)),
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "fingertips_inside": int(inside),
                    "min_tip_dist_px": float(min_dist)
                })

    return contacts