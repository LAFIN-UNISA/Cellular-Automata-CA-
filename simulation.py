# simulation.py
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from parameters import SimulationParams
from update import update_step

Position = Tuple[int, int]

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
    seed: Optional[int] = None,
    **lane_change_params
) -> Dict[str, Any]:

    if seed is not None:
        np.random.seed(seed)

    params = SimulationParams()

    # -------------------------------
    # Initial state
    # -------------------------------
    road = np.zeros((n_lanes, params.road.length), dtype=int)
    v_pos, v_speeds, v_cooldowns, v_types = {}, {}, {}, {}

    labels = list(fleet_mix.keys())
    probs = np.array([fleet_mix[k] for k in labels])
    probs /= probs.sum()

    n_vehicles = int(density * n_lanes * params.road.length)
    vid = 1
    while len(v_pos) < n_vehicles:
        lane = np.random.randint(n_lanes)
        col = np.random.randint(params.road.length)
        if road[lane, col] == 0:
            vtype = np.random.choice(labels, p=probs)
            road[lane, col] = vid
            v_pos[vid] = (lane, col)
            v_speeds[vid] = np.random.randint(speed_limits[vtype]["v_max"] + 1)
            v_cooldowns[vid] = 0
            v_types[vid] = vtype
            vid += 1

    # -------------------------------
    # Histories (for GIF + acoustics)
    # -------------------------------
    road_history = []
    positions_history = []
    speeds_history = []
    light_history = []

    total_moved = 0
    total_lane_changes = 0

    # ===============================
    #   TIME LOOP
    # ===============================
    for t in range(sim_steps):

        road_history.append(road.copy())
        positions_history.append(dict(v_pos))
        speeds_history.append(dict(v_speeds))

        light_history.append(
            "R" if (use_traffic_light and params.light.is_red(t)) else "G"
        )

        road, v_speeds, v_cooldowns, v_pos, moved, lane_changes = update_step(
            road,
            v_speeds,
            v_cooldowns,
            v_pos,
            v_types,
            speed_limits,
            frame=t,
            params=params,
            use_traffic_light=use_traffic_light,
            use_changing_lanes=use_changing_lanes,
            **lane_change_params
        )

        total_moved += moved
        total_lane_changes += lane_changes

    return {
        "road_history": road_history,
        "positions_history": positions_history,
        "speeds_history": speeds_history,
        "vehicle_types": v_types,
        "light_state_history": light_history,
        "mean_flow": total_moved / sim_steps,
        "total_lane_changes": total_lane_changes,
        "n_vehicles": n_vehicles,
    }