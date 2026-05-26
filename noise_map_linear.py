# noise_map_linear.py

import numpy as np


def propagate_Lw_to_Lp_linear(
    Lw,
    r,
    r0=1.0
):
    """
    Propagazione lineare da Lw a Lp secondo:
        Lp = Lw - 20*log10(r/r0) - 11

    Parametri
    ----------
    Lw : float
        Livello di potenza sonora [dB]
    r : float or ndarray
        Distanza sorgente-ricevitore [m]
    r0 : float
        Distanza di riferimento [m]

    Ritorna
    -------
    Lp : float or ndarray
        Livello di pressione sonora [dB]
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
    Crea la griglia spaziale (x, y) per la mappa di rumore.
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
    Calcola il campo di livello Lp(x,y) per un frame temporale,
    sommando i contributi logaritmicamente.

    La posizione laterale dei veicoli è letta direttamente da df["y_m"].
    """

    # accumulatore energetico
    S = np.zeros_like(X)

    for _, row in df_frame.iterrows():

        x_i = row["x_m"]
        y_i = row["y_m"]          # ← ORA USIAMO QUESTO
        Lw_i = row["Lw_total"]

        # distanza euclidea
        # r = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
        r_raw = np.sqrt((X - x_i) ** 2 + (Y - y_i) ** 2)
        r = np.maximum(r_raw, r_min)   # cutoff di campo vicino


        # propagazione
        Lp_i = propagate_Lw_to_Lp_linear(Lw_i, r)

        # somma logaritmica (energetica)
        S += 10.0 ** (Lp_i / 10.0)

    # ritorno in dB
    Lp_tot = 10.0 * np.log10(S)

    return Lp_tot
