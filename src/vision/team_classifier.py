from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_jersey_color(frame, row):
    """
    Extract a representative jersey color from a player's
    bounding box.

    We intentionally use the upper-middle portion of the box:
    - avoids grass near the feet
    - avoids much of the head/hair
    - focuses mainly on the shirt
    """

    frame_h, frame_w = frame.shape[:2]

    x = int(row.x)
    y = int(row.y)
    w = int(row.width)
    h = int(row.height)

    # Torso region inside bounding box.
    x1 = int(x + 0.20 * w)
    x2 = int(x + 0.80 * w)

    y1 = int(y + 0.20 * h)
    y2 = int(y + 0.60 * h)

    # Clamp to frame.
    x1 = max(0, min(x1, frame_w - 1))
    x2 = max(0, min(x2, frame_w))

    y1 = max(0, min(y1, frame_h - 1))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Convert BGR -> LAB.
    #
    # LAB is generally better than raw RGB/BGR for clustering
    # perceptual colors under changing illumination.
    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB,
    )

    pixels = lab.reshape(-1, 3)

    # ---------------------------------------------------------
    # Remove likely grass pixels.
    # ---------------------------------------------------------
    #
    # Convert crop to HSV just for grass filtering.

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    hsv_pixels = hsv.reshape(-1, 3)

    # Typical green range.
    grass_mask = (
        (hsv_pixels[:, 0] >= 30)
        & (hsv_pixels[:, 0] <= 95)
        & (hsv_pixels[:, 1] >= 40)
    )

    pixels = pixels[~grass_mask]

    if len(pixels) < 10:
        return None

    # Median is more resistant to:
    # skin, shorts, shadows, numbers, etc.
    return np.median(
        pixels,
        axis=0,
    )


def infer_goalkeepers(mot):
    """Fallback role hint: persistent extreme depth, rather than shirt color.

    Uses median pitch x when calibrated coordinates exist. Image foot x is
    a weaker backup. These are candidates; authoritative GSR/metadata wins.
    """
    column = 'pitch_x' if 'pitch_x' in mot else 'foot_x'
    depth = mot.groupby('player_id')[column].median().dropna()
    if len(depth) < 4:
        return {}
    return {int(depth.idxmin()): 'left', int(depth.idxmax()): 'right'}


def cluster_team_colors(colors, depth, goalkeepers):
    """Cluster outfield jerseys only; give each keeper the defending side's team."""
    player_ids = sorted(pid for pid in colors if pid not in goalkeepers)
    if len(player_ids) < 2:
        raise RuntimeError('Not enough outfield jersey observations')
    samples = np.asarray([colors[p] for p in player_ids], dtype=np.float32)
    model = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = model.fit_predict(samples)
    centers = np.uint8(np.clip(model.cluster_centers_,0,255)).reshape(1,2,3)
    bgr = cv2.cvtColor(centers, cv2.COLOR_LAB2BGR)[0].astype(float)
    blue_cluster = int(np.argmax(bgr[:,0]-bgr[:,2]))
    assignments = {pid:0 if int(label)==blue_cluster else 1 for pid,label in zip(player_ids,labels)}
    if goalkeepers:
        group_depth = {team:np.median([depth[p] for p in assignments if assignments[p]==team and p in depth])
                       for team in (0,1)}
        if not np.isfinite(list(group_depth.values())).all() or group_depth[0] == group_depth[1]:
            raise RuntimeError('Cannot associate goalkeeper sides with outfield teams')
        left_team = min(group_depth,key=group_depth.get)
        for pid,side in goalkeepers.items():
            assignments[pid] = left_team if side=='left' else 1-left_team
    return assignments


def classify_teams(video_path, mot, sample_every=50):
    """Fallback classifier when authoritative team labels are unavailable.

    Keeper candidates are inferred from persistent backfield position and
    excluded from the jersey clusters because their kits differ from teammates.
    """
    if sample_every < 1:
        raise ValueError('sample_every must be positive')
    keepers = infer_goalkeepers(mot)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    by_frame = {int(n):rows for n,rows in mot.groupby('frame')}
    observations = defaultdict(list)
    try:
        for frame_number in range(int(mot.frame.min()),int(mot.frame.max())+1,sample_every):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ok,frame = cap.read()
            if not ok:
                continue
            players = by_frame.get(frame_number)
            if players is None:
                continue
            for row in players.itertuples():
                if int(row.player_id) in keepers:
                    continue
                color = extract_jersey_color(frame,row)
                if color is not None:
                    observations[int(row.player_id)].append(color)
    finally:
        cap.release()
    colors = {p:np.median(values,axis=0) for p,values in observations.items()}
    column = 'pitch_x' if 'pitch_x' in mot else 'foot_x'
    depth = mot.groupby('player_id')[column].median().dropna().to_dict()
    return cluster_team_colors(colors,depth,keepers)
