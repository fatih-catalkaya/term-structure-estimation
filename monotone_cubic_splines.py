import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.interpolate import pchip_interpolate


def compute_monotone_cubic_splines(df: pd.DataFrame, today: datetime) -> None:
    unique_maturities = df["maturity"].unique()

    maturities = df["maturity"]
    ytm = maturities.apply(lambda row: (row - today).days / 365.0)
    plt.scatter(ytm, df["yield"], marker="x", label="German bonds", color="tab:orange")

    yields = []
    for maturity in unique_maturities:
        bonds_for_maturity = df[df["maturity"] == maturity]
        yields.append(np.mean(bonds_for_maturity["yield"]))
    unique_ytm = [(row - today).days / 365.0 for row in unique_maturities]
    plot_x_coords = np.linspace(min(unique_ytm), max(unique_ytm), num=1000)
    y_monotonic_cubic_spline = pchip_interpolate(unique_ytm, yields, plot_x_coords)
    plt.plot(plot_x_coords, y_monotonic_cubic_spline)
    plt.show()
