# simulation.py
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from parameters import SimulationParams
from update import update_step

Position = Tuple[int, int]

# ==================================================
#   VEHICLE SPAWNING HELPER
# ==================================================
def _spawn_vehicles(road, density, fleet_mix, speed_limits, length_map, road_id, vid_start, lane_directions=None):
    """Spawn vehicles randomly on a given road array. Returns updated dicts."""
    n_lanes, length = road.shape
    labels = list(fleet_mix.keys())
    probs = np.array([fleet_mix[k] for k in labels])
    probs /= probs.sum()

    # Density = fraction of cells occupied
    # Average vehicle length weighted by fleet_mix
    avg_len = sum(
        fleet_mix[vt] * length_map.get(vt, 1)
        for vt in fleet_mix
    ) / sum(fleet_mix.values())
    # Number of vehicles to reach target cell occupancy
    n_vehicles = int((density * n_lanes * length) / avg_len)

    v_pos, v_speeds, v_cooldowns, v_types, v_road_id = {}, {}, {}, {}, {}
    vid = vid_start

    attempts = 0
    while len(v_pos) < n_vehicles and attempts < n_vehicles * 10:
        attempts += 1
        lane  = np.random.randint(n_lanes)
        col   = np.random.randint(length)
        vtype = np.random.choice(labels, p=probs)
        vlen  = length_map.get(vtype, 1)
        # Determine lane travel direction (default +1)
        direction = 1
        if lane_directions is not None:
            direction = lane_directions.get((road_id, lane), 1)
        # Check head cell AND all body cells are free (body extends opposite to travel)
        cells = [(col - b * direction) % length for b in range(vlen)]
        if any(road[lane, c] != 0 for c in cells):
            continue
        # Mark ALL body cells on grid to block future spawns
        for c in cells:
            road[lane, c] = vid
        v_pos[vid] = (lane, col)
        v_speeds[vid] = np.random.randint(speed_limits[vtype]["v_max"] + 1)
        v_cooldowns[vid] = 0
        v_types[vid] = vtype
        v_road_id[vid] = road_id
        vid += 1

    return v_pos, v_speeds, v_cooldowns, v_types, v_road_id, vid


# ==================================================
#   MAIN SIMULATION
# ==================================================
def run_simulation(
    density: float,
    n_lanes: int,
    sim_steps: int,
    *,
    fleet_mix: Dict[str, float],
    length_map: Dict[str, int],
    speed_limits: Dict[str, Dict[str, int]],
    use_traffic_light: bool = False,
    use_changing_lanes: bool = False,
    use_intersection: bool = False,
    turning_ratio: float = 0.5,
    bidirectional: bool = False,
    priority_look_ahead: int = 15,
    seed: Optional[int] = None,
    **lane_change_params
) -> Dict[str, Any]:

    if seed is not None:
        np.random.seed(seed)

    params = SimulationParams()
    params.intersection.use_intersection = use_intersection
    params.intersection.turning_ratio = turning_ratio
    params.intersection.priority_look_ahead = priority_look_ahead
    params.intersection.bidirectional = bidirectional

    # -------------------------------
    # Road initialisation
    # horizontal road: vehicles move right  (road_id = 0)
    # vertical road:   vehicles move up     (road_id = 1)
    # -------------------------------
    road_h = np.zeros((n_lanes, params.road.length), dtype=int)  # horizontal
    road_v = np.zeros((n_lanes, params.road.length), dtype=int)  # vertical

    # Precompute lane directions so spawning marks vehicle bodies correctly
    # Assign lane directions. For right-hand traffic we invert the previous convention
    # so lanes in the first half travel with direction -1 and the second half +1.
    lane_directions = {}
    if bidirectional:
        half = n_lanes // 2
        for rid in (0, 1):
            for lane in range(n_lanes):
                lane_directions[(rid, lane)] = -1 if lane < half else 1
    else:
        for rid in (0, 1):
            for lane in range(n_lanes):
                lane_directions[(rid, lane)] = -1

    # Spawn on horizontal road (road_id = 0)
    p0, s0, c0, t0, r0, next_vid = _spawn_vehicles(
        road_h, density, fleet_mix, speed_limits, length_map, road_id=0, vid_start=1, lane_directions=lane_directions
    )

    # Spawn on vertical road (road_id = 1) only if intersection active
    if use_intersection:
        p1, s1, c1, t1, r1, next_vid = _spawn_vehicles(
            road_v, density, fleet_mix, speed_limits, length_map, road_id=1, vid_start=next_vid, lane_directions=lane_directions
        )
    else:
        p1, s1, c1, t1, r1 = {}, {}, {}, {}, {}

    # Merge all vehicle dicts
    v_pos       = {**p0, **p1}
    v_speeds    = {**s0, **s1}
    v_cooldowns = {**c0, **c1}
    v_types     = {**t0, **t1}
    v_road_id   = {**r0, **r1}

    # v_directions follow lane assignment
    v_directions = {vid: lane_directions.get((v_road_id[vid], v_pos[vid][0]), 1) for vid in v_pos}

    v_exit_direction = {}
    # Track whether vehicle has already decided to turn (avoid re-rolling every frame)
    v_will_turn = {}
    v_turn_types = {}
    v_turn_phases = {}
    v_turn_cells = {}

    # -------------------------------
    # Histories
    # -------------------------------
    road_h_history = []
    road_v_history = []
    positions_history = []
    speeds_history = []
    roads_history = []
    directions_history = []
    light_history = []

    total_moved = 0
    total_lane_changes = 0

    # ===============================
    #   TIME LOOP
    # ===============================
    for t in range(sim_steps):
        road_h_history.append(road_h.copy())
        road_v_history.append(road_v.copy())
        positions_history.append(dict(v_pos))
        speeds_history.append(dict(v_speeds))
        roads_history.append(dict(v_road_id))
        directions_history.append(dict(v_directions))
        light_history.append(
            "R" if (use_traffic_light and params.light.is_red(t)) else "G"
        )

        road_h, road_v, v_speeds, v_cooldowns, v_pos, v_road_id, v_will_turn, v_directions, v_exit_direction, v_turn_types, v_turn_phases, v_turn_cells, moved, lc = update_step(
            road_h=road_h,
            road_v=road_v,
            speeds=v_speeds,
            cooldowns=v_cooldowns,
            positions=v_pos,
            v_road_id=v_road_id,
            v_will_turn=v_will_turn,
            v_directions=v_directions,
            v_exit_direction=v_exit_direction,
            v_turn_types=v_turn_types,
            v_turn_phases=v_turn_phases,
            v_turn_cells=v_turn_cells,
            v_types=v_types,
            speed_limits=speed_limits,
            length_map=length_map,
            frame=t,
            params=params,
            use_traffic_light=use_traffic_light,
            use_changing_lanes=use_changing_lanes,
            lane_directions=lane_directions,
            **lane_change_params
        )

        total_moved += moved
        total_lane_changes += lc

    return {
        "road_history":      road_h_history,   # horizontal (main display)
        "road_v_history":    road_v_history,    # vertical
        "positions_history": positions_history,
        "speeds_history":    speeds_history,
        "roads_history":     roads_history,
        "vehicle_types":     v_types,
        "vehicle_roads":     v_road_id,
        "vehicle_directions": v_directions,
        "vehicle_turn_types": v_turn_types,
        "lane_directions": lane_directions,
        "directions_history": directions_history,
        "light_state_history": light_history,
        "mean_flow":         total_moved / sim_steps,
        "total_lane_changes": total_lane_changes,
        "n_vehicles":        len(v_pos),
        "use_intersection":  use_intersection,
        "use_traffic_light": use_traffic_light,
        "params":            params,
    }
