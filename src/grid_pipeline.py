from typing import Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pipeline import PipelineComponent
import matplotlib.pyplot as plt
import numpy as np


class GridPipeline(PipelineComponent):
    fig: Figure
    ax: Axes
    latest_value: Any

    def __init__(self, figsize: tuple[int, int] = (10, 8)) -> None:
        self.latest_value = None

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_zlabel("Z Coordinate")  # pyright: ignore
        self.ax.set_title("3D plot")
        self.ax.view_init(elev=90, azim=90)  # pyright: ignore
        self.ax.dist = 8  # pyright: ignore

        self.ax.set_xlim(-0.75, 0.75)
        self.ax.set_ylim(-1, 0)
        self.ax.set_zlim(-0.75, 0.75)  # pyright: ignore

        plt.ion()
        plt.show()

        return

    def process(self, data: Any) -> None:
        self.latest_value = data
        if not isinstance(data, np.ndarray):
            return

        results = data[-1]
        self.ax.clear()

        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_zlabel("Z Coordinate")  # pyright: ignore
        self.ax.set_title("3D Live Plot")
        # self.ax.view_init(elev=30, azim=45)  # pyright: ignore

        self.ax.set_xlim(-0.75, 0.75)
        self.ax.set_ylim(-1, 0)
        self.ax.set_zlim(-0.75, 0.75)  # pyright: ignore

        x_vals = results[:, 0]
        y_vals = results[:, 1]
        z_vals = results[:, 2]

        self.ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o")

        pose_connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

        right_hand_connections = [
            (start + 7, end + 7)
            for start, end in [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),  # Thumb
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),  # Index Finger
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),  # Middle Finger
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),  # Ring Finger
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),  # Pinky Finger
                (5, 9),
                (9, 13),
                (13, 17),  # Palm connections
            ]
        ]

        left_hand_connections = [
            (start + 28, end + 28)
            for start, end in [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),  # Thumb
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),  # Index Finger
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),  # Middle Finger
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),  # Ring Finger
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),  # Pinky Finger
                (5, 9),
                (9, 13),
                (13, 17),  # Palm connections
            ]
        ]

        connections = [
            *pose_connections,
            *right_hand_connections,
            *left_hand_connections,
        ]

        for start, end in connections:
            x_line = [x_vals[start], x_vals[end]]
            y_line = [y_vals[start], y_vals[end]]
            z_line = [z_vals[start], z_vals[end]]
            self.ax.plot(x_line, y_line, z_line, "r-", linewidth=2)

        self.fig.canvas.draw()
        plt.pause(0.01)
        return

    def get_data(self) -> Any:
        return self.latest_value
