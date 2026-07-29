# parameters.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

# ==================================================
#   DOMAIN / TOPOLOGY PARAMETERS
# ==================================================
@dataclass
class RoadParams:
    length: int = 100          # road length in cells (wrap-around)
    light_position_ratio: float = 0.5  # traffic light position ratio (0–1)

    def __post_init__(self):
        self.light_col = int(self.length * self.light_position_ratio)


# ==================================================
#   INTERSECTION PARAMETERS
# ==================================================
@dataclass
class IntersectionParams:
    use_intersection: bool = False     # activate intersection simulation
    intersection_col: int = 50         # column (x) of intersection on horizontal road
    intersection_row: int = 50         # row (y) of intersection on vertical road
    turning_ratio: float = 0.5         # fraction of vehicles that turn at intersection (0-1)
    conflict_zone: int = 1             # cells around intersection where conflict is checked
    priority_look_ahead: int = 1      # cells to look ahead to detect incoming vehicles (configurable priority rule)
    bidirectional: bool = False        # True = right-of-way rule, False = vertical always priority


# ==================================================
#   TRAFFIC (CELLULAR AUTOMATON) PARAMETERS
# ==================================================
@dataclass
class TrafficParams:
    p_slow: float = 0.05       # random slow-down probability (Nagel-Schreckenberg model)
    min_safe_distance: int = 3  # safety distance for lane changes


# ==================================================
#   TRAFFIC LIGHT PARAMETERS
# ==================================================
@dataclass
class TrafficLightParams:
    cycle: int = 30            # cycle duration [frames]
    RG_ratio: float = 0.5      # green / cycle ratio

    def is_red_horizontal(self, frame: int) -> bool:
        """Horizontal road: red when vertical has green (offset by half cycle)"""
        green_len = int(round(self.cycle * self.RG_ratio))
        return (frame % self.cycle) >= green_len

    def is_red_vertical(self, frame: int) -> bool:
        """Vertical road: opposite phase from horizontal"""
        return not self.is_red_horizontal(frame)

    # Keep backward compat
    def is_red(self, frame: int) -> bool:
        return self.is_red_horizontal(frame)


# ==================================================
#   GLOBAL SIMULATION PARAMETERS
# ==================================================
@dataclass
class SimulationParams:
    road: RoadParams = field(default_factory=RoadParams)
    traffic: TrafficParams = field(default_factory=TrafficParams)
    light: TrafficLightParams = field(default_factory=TrafficLightParams)
    intersection: IntersectionParams = field(default_factory=IntersectionParams)
