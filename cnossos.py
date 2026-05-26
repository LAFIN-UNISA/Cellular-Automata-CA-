# cnossos.py
import numpy as np

CNOSSOS_COEFFS = {
    "TW": {"A0": 84.0, "A1": 25.0, "A2": 0.0, "B0": 92.0, "B1": 20.0, "B2": 3.0},
    "L":  {"A0": 90.0, "A1": 32.0, "A2": 1.0, "B0": 98.0, "B1": 18.0, "B2": 1.5},
    "M":  {"A0": 95.0, "A1": 28.0, "A2": 1.0, "B0":102.0, "B1": 16.0, "B2": 1.5},
    "H":  {"A0":100.0, "A1": 26.0, "A2": 2.0, "B0":108.0, "B1": 15.0, "B2": 2.0},
}

V_REF = 70.0  # km/h
R0 = 1.0      # m


def lw_vehicle_mono(v_type, speed_kmh, acc_ms2):
    """
    Monoband CNOSSOS emission.
    """
    c = CNOSSOS_COEFFS[v_type]
    v = max(speed_kmh, 1e-3)

    Lw_roll = (
        c["A0"] +
        c["A1"] * np.log10(v / V_REF) +
        c["A2"] * (v - V_REF) / V_REF
    )

    Lw_engine = (
        c["B0"] +
        c["B1"] * (v - V_REF) / V_REF +
        c["B2"] * acc_ms2
    )

    Lw_total = 10 * np.log10(
        10**(Lw_roll / 10) + 10**(Lw_engine / 10)
    )

    return Lw_total, Lw_roll, Lw_engine


def propagate_Lp(Lw, r, r0=R0):
    """
    Free-field hemispherical propagation.
    """
    r_eff = max(r, 1e-3)
    return Lw - 20 * np.log10(r_eff / r0) - 8