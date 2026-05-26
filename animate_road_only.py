# animate_road_only.py
# -*- coding: utf-8 -*-

"""
Animazione minimale della strada:
- solo traffico
- nessun grafico
- nessuna statistica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple

Position = Tuple[int, int]


# ============================================
# COSTRUZIONE IMMAGINE RGB DELLA STRADA
# ============================================
def build_display_grid(
    road: np.ndarray,
    vehicle_types: Dict[int, str],
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]]
) -> np.ndarray:

    n_lanes, length = road.shape
    img = np.zeros((n_lanes, length, 3))

    for lane in range(n_lanes):
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


# ============================================
# ANIMAZIONE SOLO STRADA
# ============================================
def animate_road_only(
    result: Dict,
    *,
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]],
    interval: int = 120,
    show_traffic_light: bool = True,
    close_figure: bool = True,
):
    """
    Crea una GIF che mostra SOLO la strada.
    """

    road_hist = result["road_history"]
    vehicle_types = result["vehicle_types"]
    light_hist = result.get("light_state_history", None)

    n_frames = len(road_hist)
    n_lanes, length = road_hist[0].shape

    fig, ax = plt.subplots(figsize=(16, 0.8 * n_lanes))

    img = ax.imshow(
        build_display_grid(road_hist[0], vehicle_types, length_map, colors),
        origin="lower",
        aspect="equal",
        interpolation="none"
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Traffic simulation", fontsize=16)

    # Semaforo (opzionale)
    if show_traffic_light and light_hist is not None:
        light_col = length // 2
        light_line = ax.axvline(
            x=light_col,
            lw=4,
            color="green"
        )
    else:
        light_line = None

    def animate(t):
        img.set_data(
            build_display_grid(
                road_hist[t],
                vehicle_types,
                length_map,
                colors
            )
        )

        if light_line is not None:
            light_line.set_color("red" if light_hist[t] == "R" else "green")

        return (img,) if light_line is None else (img, light_line)

    ani = FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=interval,
        blit=False
    )

    if close_figure:
        plt.close(fig)

    return ani, fig
