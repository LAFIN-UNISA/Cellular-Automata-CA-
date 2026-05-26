# update.py
# from typing import Dict, Tuple
# import numpy as np

# Position = Tuple[int, int]

# # ==================================================
# def update_step(
#     road: np.ndarray,
#     speeds: Dict[int, int],
#     cooldowns: Dict[int, int],
#     positions: Dict[int, Position],
#     v_types: Dict[int, str],
#     speed_limits: Dict[str, Dict[str, int]],
#     frame: int,
#     params,
#     *,
#     use_traffic_light: bool,
#     use_changing_lanes: bool,
#     LC_MIN_GAIN: int = 1,
#     COOL_DOWN: int = 2,
# ) -> Tuple:

#     n_lanes, length = road.shape
#     moved = 0
#     lane_changes = 0

#     vids = list(positions.keys())
#     np.random.shuffle(vids)

#     for vid in vids:
#         lane, pos = positions[vid]
#         vtype = v_types[vid]
#         v_max = speed_limits[vtype]["v_max"]

#         # 1) Acceleration
#         v = min(speeds[vid] + 1, v_max)

#         # 2) Traffic light constraint
#         if use_traffic_light and params.light.is_red(frame):
#             dist = (params.road.light_col - pos) % length
#             v = min(v, max(dist - 1, 0))

#         # 3) Gap constraint
#         gap = 0
#         for d in range(1, length):
#             if road[lane, (pos + d) % length] != 0:
#                 break
#             gap += 1
#         v = min(v, gap)
        
#         # ==================================================
#         # 4) RANDOM SLOW-DOWN (NaSch)
#         # ==================================================
#         # Implementazione fedele del modello NaSch:
#         # con probabilità p_slow si riduce la velocità di 1 cella
#         if params.traffic.p_slow > 0.0:
#             if np.random.rand() < params.traffic.p_slow:
#                 v = max(v - 1, 0)

#         # 5) Movement
#         new_pos = (pos + v) % length
#         road[lane, pos] = 0
#         road[lane, new_pos] = vid

#         positions[vid] = (lane, new_pos)
#         speeds[vid] = v
#         if v > 0:
#             moved += 1

#     return road, speeds, cooldowns, positions, moved, lane_changes




# update.py
from typing import Dict, Tuple
import numpy as np

Position = Tuple[int, int]

# ==================================================
def update_step(
    road: np.ndarray,
    speeds: Dict[int, int],
    cooldowns: Dict[int, int],
    positions: Dict[int, Position],
    v_types: Dict[int, str],
    speed_limits: Dict[str, Dict[str, int]],
    frame: int,
    params,
    *,
    use_traffic_light: bool,
    use_changing_lanes: bool,
    LC_MIN_GAIN: int = 1,
    COOL_DOWN: int = 2,
) -> Tuple:

    n_lanes, length = road.shape
    moved = 0
    lane_changes = 0

    vids = list(positions.keys())
    np.random.shuffle(vids)

    # ------------------------------------------------
    # helper: gap avanti su una corsia
    # ------------------------------------------------
    def forward_gap(lane, pos):
        gap = 0
        for d in range(1, length):
            if road[lane, (pos + d) % length] != 0:
                break
            gap += 1
        return gap

    # ------------------------------------------------
    # helper: sicurezza dietro
    # ------------------------------------------------
    def rear_safe(lane, pos):
        for d in range(1, params.traffic.m_dist + 1):
            if road[lane, (pos - d) % length] != 0:
                return False
        return True

    # =================================================
    # LOOP VEICOLI
    # =================================================
    for vid in vids:
        lane, pos = positions[vid]
        vtype = v_types[vid]
        v_max = speed_limits[vtype]["v_max"]

        # cooldown
        if cooldowns.get(vid, 0) > 0:
            cooldowns[vid] -= 1

        # ---------------------------------------------
        # 1) accelerazione
        # ---------------------------------------------
        v = min(speeds[vid] + 1, v_max)

        # ---------------------------------------------
        # 2) semaforo
        # ---------------------------------------------
        if use_traffic_light and params.light.is_red(frame):
            dist = (params.road.light_col - pos) % length
            v = min(v, max(dist - 1, 0))

        # ---------------------------------------------
        # 3) gap sulla corsia corrente
        # ---------------------------------------------
        gap_curr = forward_gap(lane, pos)
        v = min(v, gap_curr)

        # ---------------------------------------------
        # 4) LANE CHANGE (complete)
        # ---------------------------------------------
        target_lane = lane

        if use_changing_lanes and cooldowns.get(vid, 0) == 0 and n_lanes > 1:

            v_expected_curr = v

            candidate_lanes = []

            # preferenza sorpasso a sinistra
            if lane < n_lanes - 1:
                candidate_lanes.append(lane + 1)

            # rientro a destra
            if lane > 0:
                candidate_lanes.append(lane - 1)

            best_gain = 0

            for ln in candidate_lanes:

                # cella accanto libera?
                if road[ln, pos] != 0:
                    continue

                # sicurezza dietro
                if not rear_safe(ln, pos):
                    continue

                gap_ln = forward_gap(ln, pos)
                v_ln = min(v_max, gap_ln)

                gain = v_ln - v_expected_curr

                if gain >= LC_MIN_GAIN and gain > best_gain:
                    target_lane = ln
                    best_gain = gain

            if target_lane != lane:
                # esegui cambio corsia
                road[lane, pos] = 0
                road[target_lane, pos] = vid
                positions[vid] = (target_lane, pos)
                cooldowns[vid] = COOL_DOWN
                lane_changes += 1
                lane = target_lane

        # ---------------------------------------------
        # 5) gap finale sulla corsia effettiva
        # ---------------------------------------------
        gap_final = forward_gap(lane, pos)
        v_final = min(v, gap_final, v_max)

        # ---------------------------------------------
        # 6) RANDOM SLOWDOWN (NaSch)
        # ---------------------------------------------
        if params.traffic.p_slow > 0:
            if np.random.rand() < params.traffic.p_slow:
                v_final = max(v_final - 1, 0)

        # ---------------------------------------------
        # 7) movimento
        # ---------------------------------------------
        new_pos = (pos + v_final) % length
        road[lane, pos] = 0
        road[lane, new_pos] = vid

        positions[vid] = (lane, new_pos)
        speeds[vid] = v_final

        if v_final > 0:
            moved += 1

    return road, speeds, cooldowns, positions, moved, lane_changes