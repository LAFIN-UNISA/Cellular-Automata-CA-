# animate_acoustics.py
# -*- coding: utf-8 -*-

"""
Animation of the total sound power level of the road
(Total Lw vs frame).
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_Lw_total(
    noise_db,
    *,
    interval: int = 120,
    ylabel: str = "Lw total [dB(A)]",
    close_figure: bool = True
):
    """
    Creates a GIF of the total Lw of the road that updates over time.

    noise_db must contain:
        noise_db["Lw_total"] = list of Lw values per frame
    """

    Lw_series = noise_db["Lw_total"]
    n_frames = len(Lw_series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xlim(0, n_frames)
    ax.set_ylim(
        min(v for v in Lw_series if v is not None) - 2,
        max(v for v in Lw_series if v is not None) + 2,
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title("Total sound power level of the road")

    line, = ax.plot([], [], lw=2, color="black")

    def animate(t):
        y = Lw_series[:t+1]
        x = range(len(y))
        line.set_data(x, y)
        return (line,)

    ani = FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=interval,
        blit=False
    )

    if close_figure:
        plt.close(fig)

    return ani,fig

