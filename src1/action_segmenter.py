import pandas as pd

def smooth_boolean_runs(series, min_run=3):
    """
    Removes very short True runs and very short False gaps.
    series: list/bool pd.Series
    """
    s = series.astype(int).tolist()
    n = len(s)
    if n == 0:
        return series

    # compress to runs
    runs = []
    i = 0
    while i < n:
        val = s[i]
        j = i
        while j < n and s[j] == val:
            j += 1
        runs.append((val, i, j))  # val, start, end(exclusive)
        i = j

    # remove short True runs
    for val, start, end in runs:
        if val == 1 and (end - start) < min_run:
            for k in range(start, end):
                s[k] = 0

    # recompute runs and fill short False gaps between True runs
    runs = []
    i = 0
    while i < n:
        val = s[i]
        j = i
        while j < n and s[j] == val:
            j += 1
        runs.append((val, i, j))
        i = j

    for idx in range(1, len(runs) - 1):
        val, start, end = runs[idx]
        prev_val = runs[idx - 1][0]
        next_val = runs[idx + 1][0]
        if val == 0 and prev_val == 1 and next_val == 1 and (end - start) < min_run:
            for k in range(start, end):
                s[k] = 1

    return pd.Series([bool(x) for x in s])

def segments_from_contact_frames(frame_df, min_len_frames=5, gap_merge_frames=3):
    """
    frame_df must have columns: frame_id(int), contact(bool)
    Returns segments with start_frame, end_frame, duration_frames.
    """
    df = frame_df.sort_values("frame_id").copy()
    df["contact"] = df["contact"].fillna(False).astype(bool)

    # smooth noise
    df["contact_smooth"] = smooth_boolean_runs(df["contact"], min_run=3)

    segments = []
    in_seg = False
    start = None
    last_true = None

    for _, r in df.iterrows():
        fid = int(r["frame_id"])
        c = bool(r["contact_smooth"])

        if c and not in_seg:
            in_seg = True
            start = fid
            last_true = fid
        elif c and in_seg:
            last_true = fid
        elif (not c) and in_seg:
            # potential end; wait for gaps? we will end now and merge later
            end = last_true
            if end is not None and (end - start + 1) >= min_len_frames:
                segments.append((start, end))
            in_seg = False
            start = None
            last_true = None

    # handle if ends with True
    if in_seg and start is not None and last_true is not None:
        end = last_true
        if (end - start + 1) >= min_len_frames:
            segments.append((start, end))

    # merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(list(seg))
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= gap_merge_frames:
                prev[1] = max(prev[1], seg[1])
            else:
                merged.append(list(seg))

    out = []
    for (s, e) in merged:
        out.append({
            "start_frame": int(s),
            "end_frame": int(e),
            "duration_frames": int(e - s + 1)
        })

    return pd.DataFrame(out)