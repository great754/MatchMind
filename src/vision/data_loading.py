from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetPaths:
    match_id: str = "118575"
    root: Path = Path("data/soccertrack")

    @property
    def sync(self):
        return self.root / 'mot' / f'{self.match_id}_sync.json'

    @property
    def bas(self):
        return self.root / 'bas' / self.match_id / f'{self.match_id}_12_class_events.json'

    def gsr_compact(self, half):
        if half not in (1, 2):
            raise ValueError('half must be 1 or 2')
        return self.root / 'gsr' / self.match_id / ('1st_compact.npz' if half == 1 else '2nd_compact.npz')

    def ball(self, half):
        if half not in (1, 2):
            raise ValueError('half must be 1 or 2')
        return self.ball_first_half if half == 1 else self.ball_second_half

    @property
    def mot(self):
        return self.root / "mot" / f"{self.match_id}.txt"

    @property
    def video(self):
        return self.root / "mot" / "clips" / f"{self.match_id}.mp4"

    @property
    def raw(self):
        return self.root / "raw" / self.match_id

    @property
    def homography(self):
        return self.raw / f"{self.match_id}_homography.npy"

    @property
    def player_nodes(self):
        return self.raw / f"{self.match_id}_player_nodes.csv"

    @property
    def keypoints(self):
        return self.raw / f"{self.match_id}_keypoints.json"

    @property
    def camera_intrinsics(self):
        return self.raw / f"{self.match_id}_camera_intrinsics.npz"

    @property
    def padding_info(self):
        return self.raw / f"{self.match_id}_padding_info.csv"

    @property
    def ball_first_half(self):
        return self.root / "ball" / f"{self.match_id}_1st_ball.npz"

    @property
    def ball_second_half(self):
        return self.root / "ball" / f"{self.match_id}_2nd_ball.npz"


def load_mot(path: Path) -> pd.DataFrame:
    columns = [
        "frame",
        "player_id",
        "x",
        "y",
        "width",
        "height",
        "unused_1",
        "unused_2",
        "unused_3",
        "unused_4",
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
    )

    # Approximate point where player's feet touch the ground.
    df["foot_x"] = df["x"] + df["width"] / 2.0
    df["foot_y"] = df["y"] + df["height"]

    return df


def load_homography(path: Path) -> np.ndarray:
    H = np.load(path)

    if H.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 homography, got {H.shape}"
        )

    return H.astype(np.float64)


def load_player_nodes(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_ball_file(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)

    return {
        key: data[key]
        for key in data.files
    }


def inspect_dataset(paths: DatasetPaths):
    print("\n=== MOT ===")
    mot = load_mot(paths.mot)

    print(mot.head())
    print("frames:", mot["frame"].nunique())
    print("players:", mot["player_id"].nunique())

    print("\n=== HOMOGRAPHY ===")
    H = load_homography(paths.homography)

    print(H)

    print("\n=== PLAYER NODES ===")
    nodes = load_player_nodes(paths.player_nodes)

    print("columns:")
    print(nodes.columns.tolist())
    print(nodes.head())

    print("\n=== BALL ===")

    for name, path in [
        ("first half", paths.ball_first_half),
        ("second half", paths.ball_second_half),
    ]:
        ball = load_ball_file(path)

        print(f"\n{name}: {path}")

        for key, value in ball.items():
            print(
                key,
                value.shape,
                value.dtype,
            )


if __name__ == "__main__":
    inspect_dataset(DatasetPaths())