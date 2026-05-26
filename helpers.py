# helpers.py
# -*- coding: utf-8 -*-

from typing import Iterable, List
import numpy as np

__all__ = [
    "forward_gap_limited",
    "rear_gap_safe",
    "moving_average",
]

# ==================================================
#   SPATIAL / TOPOLOGICAL HELPERS
# ==================================================
def forward_gap_limited(
    road: np.ndarray,
    lane: int,
    col: int,
    max_search: int,
    road_length: int
) -> int:
    """
    Numero di celle libere davanti al veicolo (wrap-around),
    limitando la ricerca a 'max_search' celle.

    NON decide velocità, NON usa v_max implicitamente.
    """
    gap = 0
    for d in range(1, max_search + 1):
        if road[lane, (col + d) % road_length] != 0:
            break
        gap += 1
    return gap


def rear_gap_safe(
    road: np.ndarray,
    lane: int,
    col: int,
    safety_dist: int,
    road_length: int
) -> bool:
    """
    Controlla che la distanza dietro sia libera per almeno
    'safety_dist' celle (wrap-around).
    """
    for d in range(1, safety_dist + 1):
        if road[lane, (col - d) % road_length] != 0:
            return False
    return True


# ==================================================
#   GENERIC NUMERICAL UTILITIES
# ==================================================
def moving_average(seq: Iterable[float], window: int) -> List"""
    Media mobile centrata.
    Utility generale, indipendente dal modello di traffico.
    """
    seq = list(seq)
    n = len(seq)
    if window <= 1 or window > n:
        return seq.copy()

    out = []
    half = window // 2
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        out.append(sum(seq[i0:i1]) / (i1 - i0))
    return out