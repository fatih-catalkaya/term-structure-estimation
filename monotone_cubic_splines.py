import math
import sys
from typing import Any

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


def compute_monotone_cubic_splines(df: pd.DataFrame, today: datetime, ax) \
        -> tuple[PchipInterpolator, list[float | Any]]:
    unique_maturities = df["maturity"].unique()

    maturities = df["maturity"]
    ytm = maturities.apply(lambda row: (row - today).days / 365.0)
    ax.scatter(ytm, df["yield"], marker="x", label="German bonds", color="tab:orange")

    yields = []
    for maturity in unique_maturities:
        bonds_for_maturity = df[df["maturity"] == maturity]
        yields.append(np.mean(bonds_for_maturity["yield"]))
    unique_ytm = [(row - today).days / 365.0 for row in unique_maturities]
    plot_x_coords = np.linspace(min(unique_ytm), max(unique_ytm), num=1000)
    pchip = PchipInterpolator(unique_ytm, yields)
    y_monotonic_cubic_spline = [pchip(y) for y in plot_x_coords]
    ax.plot(plot_x_coords, y_monotonic_cubic_spline)
    return pchip, unique_ytm


def interest_rate_to_discount_factor(interest_rate: float, maturity) -> float:
    return math.exp(-interest_rate * maturity)


def compute_forward_curve(pchip_interpolator: PchipInterpolator, ytm_min: float, ytm_max: float, ax) -> None:
    h = math.sqrt(sys.float_info.epsilon)
    inst_f_rates = []
    ytms = np.linspace(ytm_min, ytm_max, num=50)
    for ytm in ytms:
        ytm_plus_h = ytm + h
        inst_f_rate_plus_h = -math.log(interest_rate_to_discount_factor(pchip_interpolator(ytm_plus_h), ytm_plus_h))
        ytm_minus_h = ytm - h
        inst_f_rate_minus_h = -math.log(interest_rate_to_discount_factor(pchip_interpolator(ytm_minus_h), ytm_minus_h))
        inst_f_rates.append((inst_f_rate_plus_h - inst_f_rate_minus_h) / (2.0 * h))

    ax.plot(ytms, inst_f_rates)
    return None


def plot_splines(df: pd.DataFrame, today: datetime) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    pchip_interpolator, unique_ytms = compute_monotone_cubic_splines(df, today, ax1)
    unique_ytms = unique_ytms
    compute_forward_curve(pchip_interpolator, min(unique_ytms), max(unique_ytms), ax2)
    fig.show()
