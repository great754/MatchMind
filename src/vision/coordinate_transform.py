import ast
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay


class PitchCoordinateTransformer:
    """
    Maps SoccerTrack panoramic image coordinates onto a
    105m x 68m pitch.

    SoccerTrack's panoramic image has nonlinear/fisheye
    distortion, so a single global homography is insufficient.

    Instead:
        1. Load SoccerTrack's supplied image <-> pitch keypoints
        2. Delaunay-triangulate them in image space
        3. Find the triangle containing each player
        4. Use barycentric interpolation to obtain pitch position
    """

    def __init__(self, keypoints_path: str | Path):

        self.keypoints_path = Path(keypoints_path)

        (
            self.image_points,
            self.pitch_points,
        ) = self._load_keypoints()

        self.triangulation = Delaunay(
            self.image_points
        )

    def _load_keypoints(self):

        with open(
            self.keypoints_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        image_points = []
        pitch_points = []

        for pitch_string, image_coordinate in data.items():

            pitch_x, pitch_y = ast.literal_eval(
                pitch_string
            )

            image_x, image_y = image_coordinate

            image_points.append(
                [float(image_x), float(image_y)]
            )

            pitch_points.append(
                [float(pitch_x), float(pitch_y)]
            )

        return (
            np.asarray(
                image_points,
                dtype=np.float64,
            ),
            np.asarray(
                pitch_points,
                dtype=np.float64,
            ),
        )

    def transform_point(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float] | None:

        point = np.array(
            [x, y],
            dtype=np.float64,
        )

        simplex = self.triangulation.find_simplex(
            point
        )

        # Outside convex hull of calibration landmarks.
        if simplex < 0:
            return None

        transform = self.triangulation.transform[
            simplex
        ]

        delta = point - transform[2]

        barycentric = np.dot(
            transform[:2],
            delta,
        )

        weights = np.concatenate(
            (
                barycentric,
                [1 - barycentric.sum()],
            )
        )

        vertex_indices = (
            self.triangulation.simplices[
                simplex
            ]
        )

        pitch_triangle = self.pitch_points[
            vertex_indices
        ]

        pitch_position = np.sum(
            pitch_triangle
            * weights[:, None],
            axis=0,
        )

        return (
            float(pitch_position[0]),
            float(pitch_position[1]),
        )

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:

        results = []

        for x, y in points:

            result = self.transform_point(
                float(x),
                float(y),
            )

            if result is None:
                results.append(
                    [np.nan, np.nan]
                )
            else:
                results.append(result)

        return np.asarray(
            results,
            dtype=np.float64,
        )

    def validation_error(self):
        """
        Leave-one-in sanity test.

        Every supplied calibration point should map essentially
        exactly to its associated pitch coordinate.
        """

        predicted = self.transform_points(
            self.image_points
        )

        errors = np.linalg.norm(
            predicted - self.pitch_points,
            axis=1,
        )

        finite = np.isfinite(errors)

        return {
            "mean_m": float(
                np.mean(errors[finite])
            ),
            "median_m": float(
                np.median(errors[finite])
            ),
            "max_m": float(
                np.max(errors[finite])
            ),
        }


if __name__ == "__main__":

    transformer = PitchCoordinateTransformer(
        "data/soccertrack/raw/118575/"
        "118575_keypoints.json"
    )

    print(
        "Calibration:",
        transformer.validation_error(),
    )

    tests = [
        (106, 598),
        (2050, 242),
        (3936, 549),
    ]

    print("\nKnown keypoint tests:")

    for point in tests:

        result = transformer.transform_point(
            *point
        )

        print(
            point,
            "->",
            result,
        )