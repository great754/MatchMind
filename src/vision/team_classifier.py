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


def classify_teams(
    video_path,
    mot,
    sample_every=50,
):
    """
    Estimate a stable jersey-color representation for each
    MOT player ID, then cluster the 22 player tracks.

    Initially uses 2 clusters:
        team_1
        team_2

    Goalkeepers can be separated afterward.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {video_path}"
        )

    mot_by_frame = {
        int(frame_number): group
        for frame_number, group
        in mot.groupby("frame")
    }

    observations = defaultdict(list)

    # Seek to evenly spaced frames instead of decoding the whole clip twice.
    try:
        for frame_number in range(int(mot.frame.min()), int(mot.frame.max()) + 1, sample_every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue
            players = mot_by_frame.get(frame_number)
            if players is None:
                continue
            for row in players.itertuples():
                color = extract_jersey_color(frame, row)
                if color is not None:
                    observations[int(row.player_id)].append(color)
    finally:
        cap.release()



    # ---------------------------------------------------------
    # Aggregate each player's observations.
    # ---------------------------------------------------------

    player_ids = []
    player_colors = []

    for player_id in sorted(
        observations.keys()
    ):

        samples = np.asarray(
            observations[player_id],
            dtype=np.float32,
        )

        if len(samples) == 0:
            continue

        # Median color across the entire 4-minute clip.
        representative = np.median(
            samples,
            axis=0,
        )

        player_ids.append(player_id)
        player_colors.append(
            representative
        )

    player_colors = np.asarray(
        player_colors,
        dtype=np.float32,
    )

    if len(player_ids) < 2:
        raise RuntimeError(
            "Not enough player color observations."
        )

    # ---------------------------------------------------------
    # Team clustering
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20,
    )

    labels = kmeans.fit_predict(
        player_colors
    )

    # Anchor cluster labels to the bluer jersey, so colors remain repeatable.
    centers = np.uint8(np.clip(kmeans.cluster_centers_, 0, 255)).reshape(1, 2, 3)
    bgr = cv2.cvtColor(centers, cv2.COLOR_LAB2BGR)[0].astype(float)
    blue_cluster = int(np.argmax(bgr[:, 0] - bgr[:, 2]))
    return {player_id: (0 if int(label) == blue_cluster else 1)
            for player_id, label in zip(player_ids, labels)}
