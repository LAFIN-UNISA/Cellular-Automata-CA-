# parameters.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

# ==================================================
#   DOMAIN / TOPOLOGY PARAMETERS
# ==================================================
@dataclass
class RoadParams:
    length: int = 100          # numero celle (wrap-around)
    light_position_ratio: float = 0.5  # posizione semaforo (0–1)

    def __post_init__(self):
        self.light_col = int(self.length * self.light_position_ratio)


# ==================================================
#   TRAFFIC (CA) PARAMETERS
# ==================================================
@dataclass
class TrafficParams:
    p_slow: float = 0.05      # random slow-down probability
    m_dist: int = 3           # distanza sicurezza lane-change


# ==================================================
#   TRAFFIC LIGHT PARAMETERS
# ==================================================
@dataclass
class TrafficLightParams:
    cycle: int = 30           # durata ciclo [frame]
    RG_ratio: float = 0.5     # verde / ciclo

    def is_red(self, frame: int) -> bool:
        green_len = int(round(self.cycle * self.RG_ratio))
        return (frame % self.cycle) >= green_len


# ==================================================
#   GLOBAL SIMULATION PARAMETERS
# ==================================================

@dataclass
class SimulationParams:
    road: RoadParams = field(default_factory=RoadParams)
    traffic: TrafficParams = field(default_factory=TrafficParams)
    light: TrafficLightParams = field(default_factory=TrafficLightParams)
