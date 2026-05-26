# noise_database.py
import numpy as np
import pandas as pd

from cnossos import lw_vehicle_mono

def build_noise_database_over_time(all_frames, cell_length_m, dt, return_df=True):
    """
    Costruisce DB temporale di emissioni a partire
    da storici microscopici di traffico.
    """

    prev_speed_ms = {}
    prev_pos = {}
    wrap_count = {}

    rows = []
    db = {"frames": [], "Lw_total": []}

    for t, fr in enumerate(all_frames):

        ids = fr["ids"]
        types = fr["types"]
        speeds_cells = fr["speeds_cells"]
        pos_cells = fr["positions_cells"]
        light_state = fr.get("light_state")

        frame_Lw = []

        for i, vid in enumerate(ids):
            vtype = types[i]
            speed_ms = speeds_cells[i] * cell_length_m / dt
            speed_kmh = speed_ms * 3.6

            acc = (speed_ms - prev_speed_ms.get(vid, speed_ms)) / dt
            prev_speed_ms[vid] = speed_ms

            lane, cell = pos_cells[i]

            if vid in prev_pos and cell < prev_pos[vid]:
                wrap_count[vid] = wrap_count.get(vid, 0) + 1
            prev_pos[vid] = cell

            Lw_tot, Lw_roll, Lw_eng = lw_vehicle_mono(vtype, speed_kmh, acc)
            frame_Lw.append(Lw_tot)

            rows.append({
                "frame": t,
                "id": vid,
                "type": vtype,
                "x_m": cell * cell_length_m,
                "y_m": lane * cell_length_m,
                "speed_ms": speed_ms,
                "speed_kmh": speed_kmh,
                "acc_ms2": acc,
                "Lw_total": Lw_tot,
                "Lw_roll": Lw_roll,
                "Lw_engine": Lw_eng,
                "light_state": light_state,
                "loop_index": wrap_count.get(vid, 0),
            })

        db["frames"].append(rows[-len(ids):])
        db["Lw_total"].append(
            None if len(frame_Lw) == 0 else
            10 * np.log10(np.sum(10**(np.array(frame_Lw)/10)))
        )

    if return_df:
        return db, pd.DataFrame(rows)
    return db