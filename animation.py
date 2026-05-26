# animation.py
# -*- coding: utf-8 -*-

"""
Visualizzazione e animazione.
Questo modulo NON simula: usa solo storici già calcolati.
"""

from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

Position = Tuple[int, int]

# ==================================================
#   DISPLAY GRID
# ==================================================
def build_display_grid(
    road: np.ndarray,
    vehicle_types: Dict[int, str],
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]]
) -> np.ndarray:

    nlanes, length = road.shape
    img = np.zeros((nlanes, length, 3))

    for lane in range(nlanes):
        for col in range(length):
            vid = road[lane, col]
            if vid == 0:
                continue
            vtype = vehicle_types[vid]
            color = colors[vtype]
            vlen = length_map[vtype]
            img[lane, col] = color
            for b in range(1, vlen):
                img[lane, (col - b) % length] = color

    return img


# ==================================================
#   MAIN ANIMATION
# ==================================================
def animate_from_history(
    history: Dict,
    *,
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]],
    interval: int = 120,
    close_figure: bool = True,
):

    road_hist = history["road_history"]
    light_hist = history["light_state_history"]
    vehicle_types = history["vehicle_types"]

    steps = len(road_hist)
    n_lanes, length = road_hist[0].shape

    # -------------------------------
    # Figure layout
    # -------------------------------
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 1, height_ratios=[4, 1], figure=fig)

    ax_road = fig.add_subplot(gs[0])
    ax_density = fig.add_subplot(gs[1])

    img = ax_road.imshow(
        build_display_grid(road_hist[0], vehicle_types, length_map, colors),
        origin="lower",
        aspect="equal",
        interpolation="none"
    )
    ax_road.set_xticks([])
    ax_road.set_yticks([])

    light_line = ax_road.axvline(
        x=length // 2,
        lw=5,
        color="green"
    )

    density_line, = ax_density.plot([], [], lw=2)
    ax_density.set_xlim(0, steps)
    ax_density.set_ylim(0, 1)
    ax_density.set_xlabel("Frame")
    ax_density.set_ylabel("Density")

    density_hist = []

    # -------------------------------
    # Animation callback
    # -------------------------------
    def animate(t):
        road = road_hist[t]
        img.set_data(
            build_display_grid(road, vehicle_types, length_map, colors)
        )

        # Semaforo
        light_line.set_color("red" if light_hist[t] == "R" else "green")

        # Densità
        density = np.count_nonzero(road) / road.size
        density_hist.append(density)
        density_line.set_data(range(len(density_hist)), density_hist)

        return img, density_line, light_line

    ani = FuncAnimation(
        fig,
        animate,
        frames=steps,
        interval=interval,
        blit=False
    )

    if close_figure:
        plt.close(fig)

    return ani, fig