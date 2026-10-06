import cv2
import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32

GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32

CENTER_CIRCLE_RADIUS = 9.15

SCALE = 12

MARGIN = 30

CANVAS_WIDTH = int(PITCH_LENGTH * SCALE) + MARGIN * 2
CANVAS_HEIGHT = int(PITCH_WIDTH * SCALE) + MARGIN * 2


def pitch_to_screen(
    x: float,
    y: float,
) -> tuple[int, int]:

    return (
        int(x * SCALE) + MARGIN,
        int(y * SCALE) + MARGIN,
    )


def create_pitch() -> np.ndarray:

    canvas = np.zeros(
        (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
        dtype=np.uint8,
    )

    # Green field
    canvas[:] = (45, 110, 45)

    white = (240, 240, 240)

    tl = pitch_to_screen(0, 0)
    br = pitch_to_screen(
        PITCH_LENGTH,
        PITCH_WIDTH,
    )

    # Boundary
    cv2.rectangle(
        canvas,
        tl,
        br,
        white,
        2,
    )

    # Halfway line
    top = pitch_to_screen(
        PITCH_LENGTH / 2,
        0,
    )

    bottom = pitch_to_screen(
        PITCH_LENGTH / 2,
        PITCH_WIDTH,
    )

    cv2.line(
        canvas,
        top,
        bottom,
        white,
        2,
    )

    # Center circle
    center = pitch_to_screen(
        PITCH_LENGTH / 2,
        PITCH_WIDTH / 2,
    )

    cv2.circle(
        canvas,
        center,
        int(CENTER_CIRCLE_RADIUS * SCALE),
        white,
        2,
    )

    cv2.circle(
        canvas,
        center,
        4,
        white,
        -1,
    )

    # Penalty areas
    penalty_top = (
        PITCH_WIDTH - PENALTY_AREA_WIDTH
    ) / 2

    penalty_bottom = (
        penalty_top + PENALTY_AREA_WIDTH
    )

    # Left
    cv2.rectangle(
        canvas,
        pitch_to_screen(0, penalty_top),
        pitch_to_screen(
            PENALTY_AREA_DEPTH,
            penalty_bottom,
        ),
        white,
        2,
    )

    # Right
    cv2.rectangle(
        canvas,
        pitch_to_screen(
            PITCH_LENGTH - PENALTY_AREA_DEPTH,
            penalty_top,
        ),
        pitch_to_screen(
            PITCH_LENGTH,
            penalty_bottom,
        ),
        white,
        2,
    )

    # Goal areas
    goal_top = (
        PITCH_WIDTH - GOAL_AREA_WIDTH
    ) / 2

    goal_bottom = (
        goal_top + GOAL_AREA_WIDTH
    )

    cv2.rectangle(
        canvas,
        pitch_to_screen(0, goal_top),
        pitch_to_screen(
            GOAL_AREA_DEPTH,
            goal_bottom,
        ),
        white,
        2,
    )

    cv2.rectangle(
        canvas,
        pitch_to_screen(
            PITCH_LENGTH - GOAL_AREA_DEPTH,
            goal_top,
        ),
        pitch_to_screen(
            PITCH_LENGTH,
            goal_bottom,
        ),
        white,
        2,
    )

    return canvas


def draw_ball(
    image,
    pitch_x,
    pitch_y,
):
    screen_x, screen_y = pitch_to_screen(
        pitch_x,
        pitch_y,
    )

    # Outer black border
    cv2.circle(
        image,
        (screen_x, screen_y),
        16,
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    # White ball
    cv2.circle(
        image,
        (screen_x, screen_y),
        5,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

def draw_player(
    canvas: np.ndarray,
    x: float,
    y: float,
    player_id: int,
    team: int | None = None,
    show_id: bool = True,
    is_goalkeeper: bool = False,
):

    sx, sy = pitch_to_screen(x, y)

    if team == 0:
        color = (255, 80, 80)

    elif team == 1:
        color = (80, 80, 255)

    else:
        color = (0, 200, 255)

    cv2.circle(
        canvas,
        (sx, sy),
        9,
        color,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (sx, sy),
        9,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if is_goalkeeper:
        # Same team color; a larger white ring and GK label identify the role.
        cv2.circle(canvas, (sx, sy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'GK', (sx + 18, sy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if not show_id:
        return

    cv2.putText(
        canvas,
        str(player_id),
        (sx + 12, sy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_ball(
    canvas: np.ndarray,
    x: float,
    y: float,
):

    sx, sy = pitch_to_screen(x, y)

    # Black outline
    cv2.circle(
        canvas,
        (sx, sy),
        16,
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    # White ball
    cv2.circle(
        canvas,
        (sx, sy),
        10,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )