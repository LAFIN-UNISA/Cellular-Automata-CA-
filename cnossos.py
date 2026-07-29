# cnossos.py
import numpy as np

# ==================================================
# CNOSSOS-EU multiband coefficients (8 octave bands)
# Frequencies: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz
# ==================================================

# Rolling noise coefficients per vehicle category
# Category 1 = L (light), 2 = M (medium), 3 = H (heavy), 4 = TW (two-wheel)
A_ROLL = {
    "L":  np.array([83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2]),
    "M":  np.array([88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6]),
    "H":  np.array([91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6]),
    "TW": np.array([0.0,  0.0,  0.0,  0.0,   0.0,   0.0,  0.0,  0.0]),
}
B_ROLL = {
    "L":  np.array([30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0]),
    "M":  np.array([30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1]),
    "H":  np.array([30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6]),
    "TW": np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
}

# Propulsion noise coefficients
A_PROP = {
    "L":  np.array([97.9,  92.5,  90.7,  87.2,  84.7,  88.0,  84.4,  77.1]),
    "M":  np.array([105.5, 100.2, 100.5,  98.7, 101.0,  97.8,  91.2,  85.0]),
    "H":  np.array([108.8, 104.2, 103.5, 102.9, 102.6,  98.5,  93.8,  87.5]),
    "TW": np.array([93.0,  93.0,  93.5,  95.3,  97.2, 100.4,  95.8,  90.9]),
}
B_PROP = {
    "L":  np.array([-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0]),
    "M":  np.array([-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5]),
    "H":  np.array([ 0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0]),
    "TW": np.array([ 4.2, 7.4, 9.8,11.6,15.7,18.9,20.3,20.6]),
}

# A-weighting corrections per octave band (63..8000 Hz)
A_WEIGHT = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])

V_REF = 70.0   # km/h reference speed


def lw_vehicle_mono(v_type: str, speed_kmh: float, acc_ms2: float):
    """
    CNOSSOS-EU multiband emission model → A-weighted monoband Lw.

    Returns (Lw_total, Lw_roll, Lw_prop) in dB(A).
    """
    v = max(speed_kmh, 1e-3)

    ar = A_ROLL[v_type]
    br = B_ROLL[v_type]
    ap = A_PROP[v_type]
    bp = B_PROP[v_type]

    # Per-band rolling noise [dB]
    Lw_roll_band  = ar + br * np.log10(v / V_REF)

    # Per-band propulsion noise [dB]
    Lw_prop_band  = ap + bp * ((v - V_REF) / V_REF)

    # Apply A-weighting
    Lw_roll_A  = Lw_roll_band  + A_WEIGHT
    Lw_prop_A  = Lw_prop_band  + A_WEIGHT

    # Sum bands logarithmically
    Lw_roll  = max(0.0, 10 * np.log10(np.sum(10 ** (Lw_roll_A  / 10))))
    Lw_prop  = max(0.0, 10 * np.log10(np.sum(10 ** (Lw_prop_A  / 10))))

    # Total emission
    Lw_total = 10 * np.log10(10 ** (Lw_roll / 10) + 10 ** (Lw_prop / 10))

    return Lw_total, Lw_roll, Lw_prop


def propagate_Lp(Lw: float, r: float, r0: float = 1.0) -> float:
    """Free-field hemispherical propagation."""
    r_eff = max(r, 1e-3)
    return Lw - 20 * np.log10(r_eff / r0) - 8