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
        """Gap to the tail of the next vehicle.
        Since all body cells are marked on the grid, stop at first foreign cell."""
        rd = get_road(road_id)
        for d in range(1, length):
            cell = (pos + d) % length
            occupant = rd[lane, cell]
            if occupant != 0 and occupant != vid:
                return d - 1
        return length - 1

    def rear_safe(lane, pos, road_id):
        rd = get_road(road_id)
        for d in range(1, params.traffic.min_safe_distance + 1):
            if rd[lane, (pos - d) % length] != 0:
                return False
        return True

    def intersection_free(exclude_vid=None):
        """Since all body cells are marked on grid, just check intersection cell."""
        for ln in range(n_lanes):
            v = road_h[ln, int_col]
            if v != 0 and v != exclude_vid:
                return False
            v = road_v[ln, int_row]
            if v != 0 and v != exclude_vid:
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

    for vid in vids:
        if vid not in positions:
            continue

        lane, pos = positions[vid]
        road_id   = v_road_id[vid]
        vtype     = v_types[vid]
        v_max     = speed_limits[vtype]["v_max"]
        vlen      = length_map.get(vtype, 1)
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
        if params.intersection.use_intersection and not use_traffic_light:
            dist_to_int = i_pos - pos  # linear, no modulo — only valid if pos < i_pos

            if 0 < dist_to_int <= v_max:
                v = min(v, 1)

                # Vertical (priority): enter only if intersection cell free
                if road_id == 1 and dist_to_int == 1:
                    if not intersection_free(exclude_vid=vid):
                        v = 0

                # Horizontal (yield): intersection free AND cell before vertical free
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
                # Clear all body cells on old lane
                for b in range(vlen):
                    get_road(road_id)[lane, (pos - b) % length] = 0
                # Place on new lane
                for b in range(vlen):
                    get_road(road_id)[tgt_lane, (pos - b) % length] = vid
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

        # 7b) Stop if turning but destination occupied
        if params.intersection.use_intersection and will_turn:
            if 0 < i_pos - pos <= conflict_zone + v_max:
                dist_to_int = i_pos - pos
                if road_id == 0 and road_v[lane, int_row] != 0:
                    v = min(v, max(dist_to_int - 1, 0))
                elif road_id == 1 and road_h[lane, int_col] != 0:
                    v = min(v, max(dist_to_int - 1, 0))

        # 8) New position
        new_pos = (pos + v) % length

        # Anti-collision guard
        rd_current = get_road(road_id)
        while v > 0:
            cell = (pos + v) % length
            if rd_current[lane, cell] == 0 or rd_current[lane, cell] == vid:
                break
            v -= 1
        new_pos = (pos + v) % length

        # 9) Move — clear all body cells, rewrite at new position
        for b in range(vlen):
            rd_current[lane, (pos - b) % length] = 0

        # Road switch: only when tail has fully crossed intersection
        switched = False
        if params.intersection.use_intersection and will_turn and v > 0:
            tail_new = new_pos - vlen + 1
            if road_id == 0 and new_pos >= i_pos and tail_new >= i_pos:
                if road_v[lane, i_pos] == 0:
                    for b in range(vlen):
                        road_v[lane, (i_pos - b) % length] = vid
                    v_road_id[vid] = 1
                    positions[vid] = (lane, i_pos)
                    v_will_turn.pop(vid, None)
                    switched = True
            elif road_id == 1 and new_pos >= i_pos and tail_new >= i_pos:
                if road_h[lane, i_pos] == 0:
                    for b in range(vlen):
                        road_h[lane, (i_pos - b) % length] = vid
                    v_road_id[vid] = 0
                    positions[vid] = (lane, i_pos)
                    v_will_turn.pop(vid, None)
                    switched = True

        if not switched:
            target_rd = get_road(v_road_id[vid])
            if target_rd[lane, new_pos] == 0 or new_pos == pos:
                for b in range(vlen):
                    target_rd[lane, (new_pos - b) % length] = vid
                positions[vid] = (lane, new_pos)
            else:
                for b in range(vlen):
                    rd_current[lane, (pos - b) % length] = vid
                positions[vid] = (lane, pos)
                v = 0

        speeds[vid] = v
        if v > 0:
            moved += 1

    return road_h, road_v, speeds, cooldowns, positions, v_road_id, v_will_turn, moved, lane_changes
