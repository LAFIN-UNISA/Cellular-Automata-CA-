# noise_database.py
import numpy as np
import pandas as pd

from cnossos import lw_vehicle_mono

INTERSECTION_CELL = 50  # must match params.intersection.intersection_col/row

def build_noise_database_over_time(all_frames, cell_length_m, dt, n_lanes=1, return_df=True):
    """
    Build temporal database of emissions from traffic microscopic histories.

    Coordinate convention:
    - Horizontal road (road_id=0): x_m = cell * cell_length_m, y_m = intersection_m
    - Vertical road   (road_id=1): x_m = intersection_m,        y_m = cell * cell_length_m
    """

    intersection_m = INTERSECTION_CELL * cell_length_m

    prev_speed_ms = {}
    prev_pos      = {}
    wrap_count    = {}

    rows = []
    db   = {"frames": [], "Lw_total": []}

    for t, fr in enumerate(all_frames):

        ids          = fr["ids"]
        types        = fr["types"]
        speeds_cells = fr["speeds_cells"]
        pos_cells    = fr["positions_cells"]
        road_ids     = fr.get("road_ids", [0] * len(ids))
        light_state  = fr.get("light_state")

        frame_Lw = []

        for i, vid in enumerate(ids):
            vtype   = types[i]
            road_id = road_ids[i]
            lane, cell = pos_cells[i]

            speed_ms  = speeds_cells[i] * cell_length_m / dt
            speed_kmh = speed_ms * 3.6
            acc       = (speed_ms - prev_speed_ms.get(vid, speed_ms)) / dt
            prev_speed_ms[vid] = speed_ms

            # Wrap detection
            if vid in prev_pos and cell < prev_pos[vid]:
                wrap_count[vid] = wrap_count.get(vid, 0) + 1
            prev_pos[vid] = cell

            Lw_tot, Lw_roll, Lw_eng = lw_vehicle_mono(vtype, speed_kmh, acc)
            frame_Lw.append(Lw_tot)

            LANE_WIDTH = 7.5  # metres per lane (visual spacing)

            # Centre lanes around intersection_m:
            # With n_lanes total, lane offsets are symmetric around 0
            # lane 0 → -LANE_WIDTH/2, lane 1 → +LANE_WIDTH/2, etc.
            lane_offset = (lane - (n_lanes - 1) / 2.0) * LANE_WIDTH

            if road_id == 0:
                x_m = cell * cell_length_m
                y_m = intersection_m + lane_offset
            else:
                x_m = intersection_m + lane_offset
                y_m = cell * cell_length_m

            rows.append({
                "frame":       t,
                "id":          vid,
                "type":        vtype,
                "road_id":     road_id,
                "x_m":         x_m,
                "y_m":         y_m,
                "speed_ms":    speed_ms,
                "speed_kmh":   speed_kmh,
                "acc_ms2":     acc,
                "Lw_total":    Lw_tot,
                "Lw_roll":     Lw_roll,
                "Lw_engine":   Lw_eng,
                "light_state": light_state,
                "loop_index":  wrap_count.get(vid, 0),
            })

        db["frames"].append(rows[-len(ids):] if ids else [])
        db["Lw_total"].append(
            None if len(frame_Lw) == 0 else
            10 * np.log10(np.sum(10 ** (np.array(frame_Lw) / 10)))
        )

    if return_df:
        return db, pd.DataFrame(rows)
    return db
