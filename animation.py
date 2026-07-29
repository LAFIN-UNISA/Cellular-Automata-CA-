# animation.py
# -*- coding: utf-8 -*-

from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

Position = Tuple[int, int]


def _directional_color(base_color, *, road_id, direction):
    color = np.array(base_color, dtype=float)
    if road_id == 0:  # horizontal road
        if direction > 0:
            return np.clip(color + np.array([0.05, 0.08, 0.02]), 0.0, 1.0)
        return np.clip(color + np.array([0.08, 0.02, 0.08]), 0.0, 1.0)
    # vertical road
    if direction > 0:
        return np.clip(color + np.array([0.02, 0.08, 0.05]), 0.0, 1.0)
    return np.clip(color + np.array([0.08, 0.04, 0.02]), 0.0, 1.0)


def build_display_grid(road, vehicle_types, length_map, colors, vehicle_directions=None):
    nlanes, length = road.shape
    img = np.zeros((nlanes, length, 3))
    directions = vehicle_directions or {}
    for lane in range(nlanes):
        for col in range(length):
            vid = road[lane, col]
            if vid == 0:
                continue
            vtype = vehicle_types.get(vid)
            if vtype is None:
                continue
            base_color = colors.get(vtype, (0.7, 0.7, 0.7))
            direction = directions.get(vid, 1)
            img[lane, col] = _directional_color(base_color, road_id=0, direction=direction)
    return img


def build_cross_grid(road_h, road_v, vehicle_types, length_map, colors, vehicle_directions=None):
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

    directions = vehicle_directions or {}

    # Horizontal vehicles — all body cells already marked on grid
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
            base_color = colors.get(vtype, (0.7, 0.7, 0.7))
            direction = directions.get(vid, 1)
            canvas[row_idx, col] = _directional_color(base_color, road_id=0, direction=direction)

    # Vertical vehicles — all body cells already marked on grid
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
            base_color = colors.get(vtype, (0.7, 0.7, 0.7))
            direction = directions.get(vid, 1)
            row_idx = length - 1 - pos
            if 0 <= row_idx < length:
                canvas[row_idx, col_idx] = _directional_color(base_color, road_id=1, direction=direction)

    return canvas


def animate_from_history(
    history: Dict,
    *,
    length_map: Dict[str, int],
    colors: Dict[str, Tuple[float, float, float]],
    interval: int = 120,
    close_figure: bool = True,
):
    road_hist        = history["road_history"]
    road_v_hist      = history.get("road_v_history")
    light_hist       = history["light_state_history"]
    vehicle_types    = history["vehicle_types"]
    use_intersection = history.get("use_intersection", False)
    use_traffic_light = history.get("use_traffic_light", False)
    params           = history.get("params")

    steps = len(road_hist)
    n_lanes, length = road_hist[0].shape

    fig, ax = plt.subplots(figsize=(10, 10) if use_intersection else (16, 0.8 * n_lanes))

    directions_hist = history.get("directions_history")
    if directions_hist is None:
        directions_hist = [history.get("vehicle_directions", {}) for _ in range(steps)]

    if use_intersection and road_v_hist is not None:
        init_img = build_cross_grid(road_hist[0], road_v_hist[0], vehicle_types, length_map, colors, directions_hist[0])
        aspect, origin = "equal", "upper"
    else:
        init_img = build_display_grid(road_hist[0], vehicle_types, length_map, colors, directions_hist[0])
        aspect, origin = "auto", "lower"

    img = ax.imshow(init_img, origin=origin, aspect=aspect, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Traffic simulation", fontsize=14)

    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # Traffic lights — only if use_traffic_light is True
    light_h, light_v = None, None
    if use_traffic_light and params is not None:
        if use_intersection:
            light_h = ax.axhline(y=length // 2, lw=3, color="green", alpha=0.5)
            light_v = ax.axvline(x=length // 2, lw=3, color="red",   alpha=0.5)
        else:
            light_h = ax.axvline(x=length // 2, lw=4, color="green")

    def animate(t):
        rh = road_hist[t]
        rv = road_v_hist[t] if (use_intersection and road_v_hist) else None

        directions = directions_hist[t] if t < len(directions_hist) else {}
        if use_intersection and rv is not None:
            frame_img = build_cross_grid(rh, rv, vehicle_types, length_map, colors, directions)
        else:
            frame_img = build_display_grid(rh, vehicle_types, length_map, colors, directions)

        img.set_data(frame_img)

        if light_h is not None and use_traffic_light and params is not None:
            if use_intersection:
                light_h.set_color("red"   if params.light.is_red_horizontal(t) else "green")
                if light_v:
                    light_v.set_color("green" if params.light.is_red_horizontal(t) else "red")
            else:
                light_h.set_color("red" if light_hist[t] == "R" else "green")

        return (img,)

    ani = FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False)

    if close_figure:
        plt.close(fig)

    return ani, fig
