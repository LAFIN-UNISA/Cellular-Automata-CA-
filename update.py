# update.py
# road_id:  0 = horizontal (→)   1 = vertical (↑)
# Priority: vertical (1) has right-of-way over horizontal (0)

from typing import Dict, Tuple
import os
import numpy as np

# Debug flag controlled via environment variable
DEBUG_INTERSECTION = os.getenv("DEBUG_INTERSECTION", "0") == "1"

RIGHT_TURN_PATTERN = [
    "00",
    "00",
    "000000",
    "111000",
    " 10",
    " 10",
]

LEFT_TURN_PATTERN = [
    "01",
    "01",
    "000100",
    "111100",
    " 00",
    " 00",
]


def _pattern_to_cells(pattern: list[str]) -> list[Tuple[int, int]]:
    cells: list[Tuple[int, int]] = []
    normalized = [row.ljust(6)[:6] for row in pattern]
    for row_idx, row in enumerate(normalized):
        for col_idx, ch in enumerate(row):
            if ch == "1":
                cells.append((row_idx, col_idx))
    return cells


RIGHT_TURN_POINTS = _pattern_to_cells(RIGHT_TURN_PATTERN)
LEFT_TURN_POINTS = _pattern_to_cells(LEFT_TURN_PATTERN)

Position = Tuple[int, int]


def update_step(
    road_h: np.ndarray,
    road_v: np.ndarray,
    speeds: Dict[int, int],
    cooldowns: Dict[int, int],
    positions: Dict[int, Position],
    v_road_id: Dict[int, int],
    v_will_turn: Dict[int, bool],
    v_directions: Dict[int, int],
    v_exit_direction: Dict[int, int],
    v_turn_types: Dict[int, str],
    v_turn_phases: Dict[int, int],
    v_turn_cells: Dict[int, Tuple[int, int]],
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
    lane_directions: Dict[Tuple[int, int], int] | None = None,
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

    def forward_gap(lane, pos, road_id, direction, vid):
        """Gap to the tail of the next vehicle along the current travel direction."""
        rd = get_road(road_id)
        for d in range(1, length):
            cell = (pos + d * direction) % length
            occupant = rd[lane, cell]
            if occupant != 0 and occupant != vid:
                return d - 1
        return length - 1

    def rear_safe(lane, pos, road_id, direction):
        rd = get_road(road_id)
        for d in range(1, params.traffic.min_safe_distance + 1):
            if rd[lane, (pos - d * direction) % length] != 0:
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

    def target_cell_free(road_id, lane, exclude_vid=None):
        if road_id == 0:
            cell_val = road_v[lane, int_row]
        else:
            cell_val = road_h[lane, int_col]
        return cell_val == 0 or cell_val == exclude_vid

    def cell_before_intersection_free(road_id, lane, direction, exclude_vid=None):
        cell_before = (int_pos(road_id) - direction) % length
        rd = get_road(road_id)
        cell_val = rd[lane, cell_before]
        return cell_val == 0 or cell_val == exclude_vid

    def has_conflicting_vehicle(road_id, lane, pos, direction, vid, positions, v_directions):
        max_dist = max(1, params.intersection.priority_look_ahead)
        # Perpendicular road id
        perp = 1 if road_id == 0 else 0
        # iterate cells in approach zone and check occupant's actual distance
        if road_id == 0:
            rd = road_v
            int_i = int_row
        else:
            rd = road_h
            int_i = int_col

        for d in range(1, max_dist + 1):
            for ln in range(n_lanes):
                # check both sides symmetrically
                for cell in ((int_i - d) % length, (int_i + d) % length):
                    occ = rd[ln, cell]
                    if occ == 0 or occ == vid:
                        continue
                    # get occupant position and direction if available
                    occ_pos = positions.get(occ)
                    occ_dir = v_directions.get(occ, 1)
                    if occ_pos is None:
                        # unknown occupant; treat as conflict conservatively
                        if DEBUG_INTERSECTION:
                            print(f"DEBUG: vid {vid} sees unknown occupant {occ} at cell {cell} on perp road")
                        return True
                    occ_col = occ_pos[1]
                    # compute occupant distance to intersection on its road
                    occ_dist = dist_to_intersection(occ_col, perp, occ_dir)
                    if 0 < occ_dist <= max_dist:
                        own_dist = dist_to_intersection(pos, road_id, direction)
                        # consider conflict only if perpendicular vehicle is closer to
                        # intersection than current vehicle, or equal distance when
                        # perpendicular road has tie-breaking priority (vertical)
                        if occ_dist < own_dist or (occ_dist == own_dist and perp == 1):
                            if DEBUG_INTERSECTION:
                                print(f"DEBUG: vid {vid} conflict with occ {occ} at dist {occ_dist} (own={own_dist})")
                            return True
                    # else occupant is not approaching within the approach zone -> ignore
        return False

    def dist_to_intersection(pos, road_id, direction):
        i = int_pos(road_id)
        if direction == 1:
            return (i - pos) % length
        return (pos - i) % length
    def steps_for_tail_to_cross(pos, vlen, road_id, direction):
        i = int_pos(road_id)
        if direction == 1:
            tail_pos = (pos - (vlen - 1)) % length
            return (i - tail_pos) % length
        else:
            tail_pos = (pos + (vlen - 1)) % length
            return (tail_pos - i) % length

    for vid in vids:
        if vid not in positions:
            continue

        lane, pos = positions[vid]
        road_id   = v_road_id[vid]
        direction = v_directions.get(vid, 1)
        if direction not in (-1, 1):
            direction = 1
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
                dist = dist_to_intersection(pos, road_id, direction)
                v = min(v, max(dist - 1, 0))
            elif road_id == 1 and params.light.is_red_vertical(frame):
                dist = dist_to_intersection(pos, road_id, direction)
                v = min(v, max(dist - 1, 0))
        elif use_traffic_light:
            dist = dist_to_intersection(pos, road_id, direction)
            v = min(v, max(dist - 1, 0))

        # 3) Intersection approach rules (no traffic light)
        if params.intersection.use_intersection and not use_traffic_light:
            dist_to_int = dist_to_intersection(pos, road_id, direction)
            turn_type = v_turn_types.get(vid, "straight")

            if 0 < dist_to_int <= v_max:
                v = min(v, 1)

                # Right: always free (horizontal has priority, right turn is safe)
                # Straight: always free (horizontal has priority)
                # Left: check that no vehicle is coming from the opposite direction
                if turn_type == "left":
                    if has_conflicting_vehicle(road_id, lane, pos, direction, vid, positions, v_directions):
                        v = 0
                    elif not target_cell_free(road_id, lane, exclude_vid=vid):
                        v = 0
        # 4) Forward gap
        gap = forward_gap(lane, pos, road_id, direction, vid)
        v   = min(v, gap)

        # 5) Random slowdown (NaSch)
        if params.traffic.p_slow > 0 and np.random.rand() < params.traffic.p_slow:
            v = max(v - 1, 0)

        # 6) Lane change
        # Restrict lane changes when approaching the intersection (priority/approach zone)
        dist_to_int_for_lc = dist_to_intersection(pos, road_id, direction)
        if params.intersection.use_intersection and not use_traffic_light and dist_to_int_for_lc <= params.intersection.priority_look_ahead:
            allow_lane_change = False
        else:
            allow_lane_change = True

        if use_changing_lanes and cooldowns.get(vid, 0) == 0 and n_lanes > 1 and allow_lane_change:
            rd        = get_road(road_id)
            best_gain = 0
            tgt_lane  = lane
            cands = []
            if lane < n_lanes - 1: cands.append(lane + 1)
            if lane > 0:           cands.append(lane - 1)
            for ln in cands:
                if rd[ln, pos] != 0: continue
                if not rear_safe(ln, pos, road_id, direction): continue
                g    = forward_gap(ln, pos, road_id, direction, vid)
                gain = min(v_max, g) - v
                if gain >= LC_MIN_GAIN and gain > best_gain:
                    tgt_lane  = ln
                    best_gain = gain
            if tgt_lane != lane:
                # Clear all body cells on old lane
                for b in range(vlen):
                    get_road(road_id)[lane, (pos - b * direction) % length] = 0
                # Place on new lane
                for b in range(vlen):
                    get_road(road_id)[tgt_lane, (pos - b * direction) % length] = vid
                positions[vid]  = (tgt_lane, pos)
                cooldowns[vid]  = COOL_DOWN
                lane_changes   += 1
                lane = tgt_lane

        # 7) Turning decision (once per approach)
        if params.intersection.use_intersection and vid not in v_will_turn:
            dist_to_int = dist_to_intersection(pos, road_id, direction)
            if 0 < dist_to_int <= conflict_zone + v_max * 2:
                if np.random.rand() < params.intersection.turning_ratio:
                    v_will_turn[vid] = True
                    turn_type = 'right' if np.random.rand() < 0.5 else 'left'
                    v_turn_types[vid] = turn_type
                    v_turn_phases[vid] = 0
                    pattern_points = RIGHT_TURN_POINTS if turn_type == 'right' else LEFT_TURN_POINTS
                    v_turn_cells[vid] = pattern_points[0] if pattern_points else (0, 0)
                    v_exit_direction[vid] = -1 if turn_type == 'right' else 1
                else:
                    v_will_turn[vid] = False

        will_turn = v_will_turn.get(vid, False)

        # 7b) Stop if turning but destination occupied
        if params.intersection.use_intersection and will_turn:
            dist_to_int = dist_to_intersection(pos, road_id, direction)
            if 0 < dist_to_int <= conflict_zone + v_max:
                if road_id == 0 and road_v[lane, int_row] != 0:
                    v = min(v, max(dist_to_int - 1, 0))
                elif road_id == 1 and road_h[lane, int_col] != 0:
                    v = min(v, max(dist_to_int - 1, 0))

        # 7c) Explicit turn trajectory phase handling near the intersection
        turn_type = v_turn_types.get(vid)
        if params.intersection.use_intersection and will_turn and turn_type in {'right', 'left'}:
            dist_to_int = dist_to_intersection(pos, road_id, direction)
            if 0 < dist_to_int <= conflict_zone + v_max:
                pattern_points = RIGHT_TURN_POINTS if turn_type == 'right' else LEFT_TURN_POINTS
                phase = v_turn_phases.get(vid, 0) + 1
                if pattern_points:
                    phase = min(phase, len(pattern_points) - 1)
                v_turn_phases[vid] = phase
                if pattern_points:
                    v_turn_cells[vid] = pattern_points[phase]
        elif turn_type in {'right', 'left'}:
            v_turn_phases.pop(vid, None)
            v_turn_cells.pop(vid, None)

        # 8) New position
        new_pos = (pos + v * direction) % length

        # Anti-collision guard
        rd_current = get_road(road_id)
        while v > 0:
            cell = (pos + v * direction) % length
            if rd_current[lane, cell] == 0 or rd_current[lane, cell] == vid:
                break
            v -= 1
        new_pos = (pos + v * direction) % length

        # 9) Move — clear all body cells, rewrite at new position
        for b in range(vlen):
            rd_current[lane, (pos - b * direction) % length] = 0

        # Road switch: only when tail has fully crossed intersection
        switched = False
        if params.intersection.use_intersection and will_turn and v > 0:
            tail_new = (new_pos - (vlen - 1) * direction) % length
            # compute required steps for tail to fully cross intersection
            needed = steps_for_tail_to_cross(pos, vlen, road_id, direction)
            if v >= needed:
                if road_id == 0:
                    # ensure destination entry cell is free
                    if road_v[lane, i_pos] == 0:
                        exit_dir = v_exit_direction.get(vid, direction)
                        pattern_points = RIGHT_TURN_POINTS if v_turn_types.get(vid) == 'right' else LEFT_TURN_POINTS
                        phase_idx = min(v_turn_phases.get(vid, 0), len(pattern_points) - 1) if pattern_points else 0
                        row_idx, col_idx = pattern_points[phase_idx] if pattern_points else (0, 0)
                        entry_pos = (i_pos + (row_idx - 2)) % length
                        new_dir = lane_directions.get((1, lane), exit_dir) if lane_directions is not None else exit_dir
                        for b in range(vlen):
                            road_v[lane, (entry_pos - b * new_dir) % length] = vid
                        v_road_id[vid] = 1
                        v_directions[vid] = new_dir
                        positions[vid] = (lane, entry_pos)
                        v_will_turn.pop(vid, None)
                        v_exit_direction.pop(vid, None)
                        v_turn_types.pop(vid, None)
                        v_turn_phases.pop(vid, None)
                        v_turn_cells.pop(vid, None)
                        switched = True
                else:
                    if road_h[lane, i_pos] == 0:
                        exit_dir = v_exit_direction.get(vid, direction)
                        pattern_points = RIGHT_TURN_POINTS if v_turn_types.get(vid) == 'right' else LEFT_TURN_POINTS
                        phase_idx = min(v_turn_phases.get(vid, 0), len(pattern_points) - 1) if pattern_points else 0
                        row_idx, col_idx = pattern_points[phase_idx] if pattern_points else (0, 0)
                        entry_pos = (i_pos + (col_idx - 2)) % length
                        new_dir = lane_directions.get((0, lane), exit_dir) if lane_directions is not None else exit_dir
                        for b in range(vlen):
                            road_h[lane, (entry_pos - b * new_dir) % length] = vid
                        v_road_id[vid] = 0
                        v_directions[vid] = new_dir
                        positions[vid] = (lane, entry_pos)
                        v_will_turn.pop(vid, None)
                        v_exit_direction.pop(vid, None)
                        v_turn_types.pop(vid, None)
                        v_turn_phases.pop(vid, None)
                        v_turn_cells.pop(vid, None)
                        switched = True

        if not switched:
            target_rd = get_road(v_road_id[vid])
            if target_rd[lane, new_pos] == 0 or new_pos == pos:
                for b in range(vlen):
                    target_rd[lane, (new_pos - b * direction) % length] = vid
                positions[vid] = (lane, new_pos)
            else:
                for b in range(vlen):
                    rd_current[lane, (pos - b * direction) % length] = vid
                positions[vid] = (lane, pos)
                v = 0

        speeds[vid] = v
        if v > 0:
            moved += 1

    return road_h, road_v, speeds, cooldowns, positions, v_road_id, v_will_turn, v_directions, v_exit_direction, v_turn_types, v_turn_phases, v_turn_cells, moved, lane_changes