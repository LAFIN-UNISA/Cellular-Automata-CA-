# animate_road_only.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple

Position = Tuple[int, int]


def build_display_grid(road, vehicle_types, length_map, colors):
    n_lanes, length = road.shape
    img = np.zeros((n_lanes, length, 3))
    for lane in range(n_lanes):
        for col in range(length):
            vid = road[lane, col]
            if vid == 0:
                continue
            vtype = vehicle_types[vid]
            img[lane, col] = colors[vtype]
            for b in range(1, length_map[vtype]):
                img[lane, (col - b) % length] = colors[vtype]
    return img


def build_cross_grid(road_h, road_v, vehicle_types, length_map, colors):
    n_lanes, length = road_h.shape
    canvas = np.full((length, length, 3), 0.15)
    mid = length // 2

    for ln in range(n_lanes):
        row_idx = mid - n_lanes // 2 + ln
        col_idx = mid - n_lanes // 2 + ln
        if 0 <= row_idx < length:
            canvas[row_idx, :] = 0.35
        if 0 <= col_idx < length:
            canvas[:, col_idx] = 0.35

    for ln in range(n_lanes):
        row_idx = mid - n_lanes // 2 + ln
        if not (0 <= row_idx < length):
            continue
        for col in range(length):
            vid = road_h[ln, col]
            if vid == 0:
                continue
            vtype = vehicle_types.get(vid)
            if vtype is None:
                continue
            canvas[row_idx, col] = colors[vtype]
            for b in range(1, length_map[vtype]):
                canvas[row_idx, (col - b) % length] = colors[vtype]

    for ln in range(n_lanes):
        col_idx = mid - n_lanes // 2 + ln
        if not (0 <= col_idx < length):
            continue
        for pos in range(length):
            vid = road_v[ln, pos]
            if vid == 0:
                continue
            vtype = vehicle_types.get(vid)
            if vtype is None:
                continue
            row_idx = length - 1 - pos
            if 0 <= row_idx < length:
                canvas[row_idx, col_idx] = colors[vtype]
            for b in range(1, length_map[vtype]):
                r = length - 1 - ((pos - b) % length)
                if 0 <= r < length:
                    canvas[r, col_idx] = colors[vtype]

    return canvas


def animate_road_only(
    result: Dict,
    *,
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]],
    interval: int = 120,
    show_traffic_light: bool = True,
    close_figure: bool = True,
):
    road_hist         = result["road_history"]
    road_v_hist       = result.get("road_v_history")
    vehicle_types     = result["vehicle_types"]
    light_hist        = result.get("light_state_history")
    use_intersection  = result.get("use_intersection", False)
    use_traffic_light = result.get("use_traffic_light", False)
    params            = result.get("params")

    n_frames = len(road_hist)
    n_lanes, length = road_hist[0].shape

    fig, ax = plt.subplots(figsize=(10, 10) if use_intersection else (16, 0.8 * n_lanes))

    if use_intersection and road_v_hist is not None:
        init_img = build_cross_grid(road_hist[0], road_v_hist[0], vehicle_types, length_map, colors)
        aspect, origin = "equal", "upper"
    else:
        init_img = build_display_grid(road_hist[0], vehicle_types, length_map, colors)
        aspect, origin = "auto", "lower"

    img = ax.imshow(init_img, origin=origin, aspect=aspect, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Traffic simulation", fontsize=14)

    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # Traffic lights — only if both show_traffic_light AND use_traffic_light are True
    light_h, light_v = None, None
    if show_traffic_light and use_traffic_light and params is not None:
        if use_intersection:
            light_h = ax.axhline(y=length // 2, lw=3, color="green", alpha=0.5)
            light_v = ax.axvline(x=length // 2, lw=3, color="red",   alpha=0.5)
        else:
            light_h = ax.axvline(x=length // 2, lw=4, color="green")

    def animate(t):
        rh = road_hist[t]
        rv = road_v_hist[t] if (use_intersection and road_v_hist) else None

        if use_intersection and rv is not None:
            frame_img = build_cross_grid(rh, rv, vehicle_types, length_map, colors)
        else:
            frame_img = build_display_grid(rh, vehicle_types, length_map, colors)

        img.set_data(frame_img)

        if light_h is not None and light_hist is not None:
            if use_intersection and params is not None:
                light_h.set_color("red"   if params.light.is_red_horizontal(t) else "green")
                if light_v:
                    light_v.set_color("green" if params.light.is_red_horizontal(t) else "red")
            else:
                light_h.set_color("red" if light_hist[t] == "R" else "green")

        return (img,)

    ani = FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=False)

    if close_figure:
        plt.close(fig)

    return ani, fig
