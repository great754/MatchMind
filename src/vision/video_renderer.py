from pathlib import Path

import cv2
import numpy as np


def overlay_tactical_view(
    video_frame: np.ndarray,
    tactical_frame: np.ndarray,
    width_ratio: float = 0.32,
    bottom_margin: int = 25,
    opacity: float = 0.88,
) -> np.ndarray:
    """
    Overlay the generated tactical pitch near the bottom-center
    of the original match footage.
    """

    output = video_frame.copy()

    frame_h, frame_w = output.shape[:2]

    # ---------------------------------------------------------
    # Resize tactical view
    # ---------------------------------------------------------

    target_width = int(frame_w * width_ratio)

    tactical_h, tactical_w = tactical_frame.shape[:2]

    scale = target_width / tactical_w

    target_height = int(tactical_h * scale)

    tactical = cv2.resize(
        tactical_frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    # ---------------------------------------------------------
    # Position: bottom-center
    # ---------------------------------------------------------

    x1 = (frame_w - target_width) // 2
    y1 = frame_h - target_height - bottom_margin

    x2 = x1 + target_width
    y2 = y1 + target_height

    # ---------------------------------------------------------
    # Slight dark backing panel
    # ---------------------------------------------------------

    padding = 8

    panel_x1 = max(0, x1 - padding)
    panel_y1 = max(0, y1 - padding)
    panel_x2 = min(frame_w, x2 + padding)
    panel_y2 = min(frame_h, y2 + padding)

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1,
    )

    output = cv2.addWeighted(
        overlay,
        0.45,
        output,
        0.55,
        0,
    )

    # ---------------------------------------------------------
    # Tactical map
    # ---------------------------------------------------------

    roi = output[y1:y2, x1:x2]

    blended = cv2.addWeighted(
        tactical,
        opacity,
        roi,
        1.0 - opacity,
        0,
    )

    output[y1:y2, x1:x2] = blended

    # ---------------------------------------------------------
    # Label
    # ---------------------------------------------------------

    cv2.putText(
        output,
        "TOP-DOWN  |  BLUE / RED",
        (x1, max(25, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


class MatchMindVideoWriter:

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Could not open output video: "
                f"{output_path}"
            )

    def write(self, frame):

        self.writer.write(frame)

    def close(self):

        self.writer.release()