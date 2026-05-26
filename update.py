# update.py
# road_id:  0 = horizontal (→)   1 = vertical (↑)
# Priority: vertical (1) has right-of-way over horizontal (0)

from typing import Dict, Tuple
import numpy as np

Position = Tuple[int, int]


def update_step(
    road_h: np.ndarray,
    road_v: np.ndarray,
    speeds: Dict[int, int],
    cooldowns: Dict[int, int],
    positions: Dict[int, Position],
    v_road_id: Dict[int, int],
    v_will_turn: Dict[int, bool],
    v_types: Dict[int, str],
    speed_limits: Dict[str, Dict[str, int]],
    length_map: Dict[str, int],
    frame: int,
    params,
    *,
    use_traffic_light: bool,
    use_changing_lanes: bool,
    LC_MIN_GAIN: int = 1,
    COOL_DOWN:   int = 2,
) -> Tuple:

    n_lanes, length = road_h.shape
    moved        = 0
    lane_changes = 0

    int_col       = params.intersection.intersection_col
    int_row       = params.intersection.intersection_row
    conflict_zone = params.intersection.conflict_zone

    # Process vertical vehicles (priority) first, then horizontal
    # This prevents two vehicles from both seeing intersection_free()=True
    # in the same frame and colliding
    vids_v = [vid for vid in positions if v_road_id.get(vid) == 1]
    vids_h = [vid for vid in positions if v_road_id.get(vid) == 0]
    np.random.shuffle(vids_v)
    np.random.shuffle(vids_h)
    vids = vids_v + vids_h

    def get_road(road_id):
        return road_v if road_id == 1 else road_h

    def int_pos(road_id):
        return int_row if road_id == 1 else int_col

    def forward_gap(lane, pos, road_id):
        rd  = get_road(road_id)
        gap = 0
        for d in range(1, length):
            if rd[lane, (pos + d) % length] != 0:
                break
            gap += 1
        return gap

    def rear_safe(lane, pos, road_id):
        rd = get_road(road_id)
        for d in range(1, params.traffic.min_safe_distance + 1):
            if rd[lane, (pos - d) % length] != 0:
                return False
        return True

    def intersection_free(exclude_vid=None):
        max_len = max(length_map.values())
        for d in range(max_len + 1):
            for ln in range(n_lanes):
                # Horizontal: vehicle head at (int_col - d) occupies
                # cells [int_col-d-vlen+1 .. int_col-d], check if int_col is in that range
                col = (int_col - d) % length
                v = road_h[ln, col]
                if v != 0 and v != exclude_vid:
                    vlen = length_map.get(v_types.get(v, "L"), 1)
                    # vehicle head at col, tail at col-vlen+1
                    # intersection at int_col: occupied if col-vlen+1 <= int_col <= col
                    if col - vlen + 1 <= int_col <= col:
                        return False
                # Vertical: same logic
                row = (int_row - d) % length
                v = road_v[ln, row]
                if v != 0 and v != exclude_vid:
                    vlen = length_map.get(v_types.get(v, "L"), 1)
                    if row - vlen + 1 <= int_row <= row:
                        return False
        return True

    def cell_before_int_v_free():
        cell_before = int_row - 1
        if cell_before < 0:
            return True
        for ln in range(n_lanes):
            if road_v[ln, cell_before] != 0:
                return False
        return True

    # ------------------------------------------------------------------
    for vid in vids:
        if vid not in positions:
            continue

        lane, pos = positions[vid]
        road_id   = v_road_id[vid]
        vtype     = v_types[vid]
        v_max     = speed_limits[vtype]["v_max"]
        i_pos     = int_pos(road_id)

        if cooldowns.get(vid, 0) > 0:
            cooldowns[vid] -= 1

        # 1) Acceleration
        v = min(speeds[vid] + 1, v_max)

        # 2) Traffic light
        if use_traffic_light and params.intersection.use_intersection:
            if road_id == 0 and params.light.is_red_horizontal(frame):
                dist = (i_pos - pos) % length
                v = min(v, max(dist - 1, 0))
            elif road_id == 1 and params.light.is_red_vertical(frame):
                dist = (i_pos - pos) % length
                v = min(v, max(dist - 1, 0))
        elif use_traffic_light:
            dist = (i_pos - pos) % length
            v = min(v, max(dist - 1, 0))

        # 3) Intersection approach rules (no traffic light)
        # Only apply to vehicles strictly BEFORE the intersection (pos < i_pos)
        # Vehicles past the intersection are never blocked.
        if params.intersection.use_intersection and not use_traffic_light:
            dist_to_int = i_pos - pos  # linear, no modulo

            if 0 < dist_to_int:
                # Everyone limited to 1 cell/frame within v_max cells of intersection
                if dist_to_int <= v_max:
                    v = min(v, 1)

                    # Vertical (priority): enter only if intersection free
                    if road_id == 1 and dist_to_int == 1:
                        if not intersection_free(exclude_vid=vid):
                            v = 0

                    # Horizontal (yield): enter only if intersection free AND cell just before intersection on vertical is free
                    elif road_id == 0 and dist_to_int == 1:
                        if not (intersection_free(exclude_vid=vid) and cell_before_int_v_free()):
                            v = 0

        # 4) Forward gap
        gap = forward_gap(lane, pos, road_id)
        v   = min(v, gap)

        # 5) Random slowdown (NaSch)
        if params.traffic.p_slow > 0 and np.random.rand() < params.traffic.p_slow:
            v = max(v - 1, 0)

        # 6) Lane change
        if use_changing_lanes and cooldowns.get(vid, 0) == 0 and n_lanes > 1:
            rd        = get_road(road_id)
            best_gain = 0
            tgt_lane  = lane
            cands = []
            if lane < n_lanes - 1: cands.append(lane + 1)
            if lane > 0:           cands.append(lane - 1)
            for ln in cands:
                if rd[ln, pos] != 0: continue
                if not rear_safe(ln, pos, road_id): continue
                g    = forward_gap(ln, pos, road_id)
                gain = min(v_max, g) - v
                if gain >= LC_MIN_GAIN and gain > best_gain:
                    tgt_lane  = ln
                    best_gain = gain
            if tgt_lane != lane:
                get_road(road_id)[lane, pos]      = 0
                get_road(road_id)[tgt_lane, pos]  = vid
                positions[vid]  = (tgt_lane, pos)
                cooldowns[vid]  = COOL_DOWN
                lane_changes   += 1
                lane = tgt_lane

        # 7) Turning decision (once per approach)
        if params.intersection.use_intersection and vid not in v_will_turn:
            dist_to_int = (i_pos - pos) % length
            if 0 < dist_to_int <= conflict_zone + v_max * 2:
                v_will_turn[vid] = (np.random.rand() < params.intersection.turning_ratio)

        will_turn = v_will_turn.get(vid, False)

        # 7b) Stop before intersection if destination occupied (turning vehicles)
        if params.intersection.use_intersection and will_turn:
            if 0 < i_pos - pos <= conflict_zone + v_max:
                dist_to_int = i_pos - pos
                if road_id == 0 and road_v[lane, int_row] != 0:
                    v = min(v, max(dist_to_int - 1, 0))
                elif road_id == 1 and road_h[lane, int_col] != 0:
                    v = min(v, max(dist_to_int - 1, 0))

        # 8) New position
        new_pos = (pos + v) % length

        # Anti-collision guard on current road
        rd_current = get_road(road_id)
        while v > 0 and rd_current[lane, new_pos] != 0 and rd_current[lane, new_pos] != vid:
            v      -= 1
            new_pos = (pos + v) % length

        # Anti-collision on destination road (turning)
        if (params.intersection.use_intersection and will_turn
                and abs(new_pos - i_pos) <= 2):
            if road_id == 0 and road_v[lane, int_row] != 0:
                v = 0; new_pos = pos
            elif road_id == 1 and road_h[lane, int_col] != 0:
                v = 0; new_pos = pos

        # 9) Move
        rd_current[lane, pos] = 0

        switched = False
        if (params.intersection.use_intersection and will_turn
                and abs(new_pos - i_pos) <= 2 and v > 0):
            if road_id == 0 and road_v[lane, int_row] == 0:
                road_v[lane, int_row] = vid
                v_road_id[vid] = 1
                positions[vid] = (lane, int_row)
                v_will_turn.pop(vid, None)
                switched = True
            elif road_id == 1 and road_h[lane, int_col] == 0:
                road_h[lane, int_col] = vid
                v_road_id[vid] = 0
                positions[vid] = (lane, int_col)
                v_will_turn.pop(vid, None)
                switched = True

        if not switched:
            target_rd = get_road(v_road_id[vid])
            if target_rd[lane, new_pos] == 0 or new_pos == pos:
                target_rd[lane, new_pos] = vid
                positions[vid] = (lane, new_pos)
            else:
                target_rd[lane, pos] = vid
                positions[vid] = (lane, pos)
                v = 0

        speeds[vid] = v
        if v > 0:
            moved += 1

    return road_h, road_v, speeds, cooldowns, positions, v_road_id, v_will_turn, moved, lane_changes
