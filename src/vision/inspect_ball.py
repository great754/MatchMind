# src/vision/inspect_ball.py

import numpy as np


def inspect(path):

    data = np.load(path)

    frame = data["frame"]
    x = data["x"]
    y = data["y"]
    status = data["status"]

    print("\n", path)

    print("Frames:")
    print(frame[0], "->", frame[-1])

    print("Count:")
    print(len(frame))

    print("\nStatus changes:")

    previous = status[0]

    for i in range(1, len(status)):

        if status[i] != previous:

            print(
                f"frame={frame[i]:6d}",
                f"time={frame[i] / 25:8.2f}s",
                f"{previous} -> {status[i]}",
                f"ball=({x[i]:.2f}, {y[i]:.2f})",
            )

            previous = status[i]


inspect(
    "data/soccertrack/ball/"
    "118575_1st_ball.npz"
)