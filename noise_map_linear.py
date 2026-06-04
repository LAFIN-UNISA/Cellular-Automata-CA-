# noise_map_linear.py

import numpy as np


def propagate_Lw_to_Lp_linear(
    Lw,
    r,
    r0=1.0
):
    """
    Linear propagation from Lw to Lp according to:
        Lp = Lw - 20*log10(r/r0) - 11

    Parameters
    ----------
    Lw : float
        Sound power level [dB]
    r : float or ndarray
        Source-receiver distance [m]
    r0 : float
        Reference distance [m]

    Returns
    -------
    Lp : float or ndarray
        Sound pressure level [dB]
    """

    r = np.maximum(r, 1e-6)
    Lp = Lw - 20.0 * np.log10(r / r0) - 11.0

    return Lp


def create_spatial_grid(
    road_length,
    dx,
    y_max,
    dy
):
    """
    Creates the spatial grid (x, y) for the noise map.
    """

    x = np.arange(0, road_length + dx, dx)
    y = np.arange(-y_max, y_max + dy, dy)

    X, Y = np.meshgrid(x, y)

    return X, Y


def compute_Lp_field_frame(
    df_frame,
    X,
    Y,
    r_min
):
    """
    Computes the Lp(x,y) sound pressure level field for a temporal frame,
    summing contributions logarithmically.

    The lateral position of vehicles is read directly from df["y_m"].
    """

    # energy accumulator
    S = np.zeros_like(X)

    for _, row in df_frame.iterrows():

        x_i = row["x_m"]
        y_i = row["y_m"]          # ← NOW WE USE THIS
        Lw_i = row["Lw_total"]

        # euclidean distance
        # r = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
        r_raw = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
        r = np.maximum(r_raw, r_min)   # near field cutoff


        # propagation
        Lp_i = propagate_Lw_to_Lp_linear(Lw_i, r)

        # logarithmic (energetic) sum
        S += 10.0 ** (Lp_i / 10.0)

    # return to dB
    Lp_tot = 10.0 * np.log10(S)

    return Lp_tot
