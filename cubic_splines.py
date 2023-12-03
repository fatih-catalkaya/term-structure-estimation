import math
import sys

import pandas as pd
import numpy as np
from datetime import datetime

from scipy.interpolate import PchipInterpolator, CubicSpline


def interest_rate_to_discount_factor(interest_rate: float, maturity) -> float:
    return math.exp(-interest_rate * maturity)


def compute_forward_curve(pchip_interpolator: PchipInterpolator, x_values) -> list[float]:
    h = math.sqrt(sys.float_info.epsilon)
    inst_f_rates = []
    for ytm in x_values:
        ytm_plus_h = ytm + h
        inst_f_rate_plus_h = -math.log(interest_rate_to_discount_factor(pchip_interpolator(ytm_plus_h), ytm_plus_h))
        ytm_minus_h = ytm - h
        inst_f_rate_minus_h = -math.log(interest_rate_to_discount_factor(pchip_interpolator(ytm_minus_h), ytm_minus_h))
        inst_f_rates.append((inst_f_rate_plus_h - inst_f_rate_minus_h) / (2.0 * h))
    return inst_f_rates


def compute_splines(df: pd.DataFrame, today: datetime) -> tuple[list[float], list[float], list[float]]:
    unique_maturities = df["maturity"].unique()
    yields = []
    for maturity in unique_maturities:
        bonds_for_maturity = df[df["maturity"] == maturity]
        yields.append(np.mean(bonds_for_maturity["yield"]))
    unique_ytm = [(row - today).days / 365.0 for row in unique_maturities]
    cubic_spline = CubicSpline(unique_ytm, yields)
    x_values = np.linspace(min(unique_ytm), max(unique_ytm), num=50).tolist()
    yield_values = [cubic_spline(x) for x in x_values]
    forward_values = compute_forward_curve(cubic_spline, x_values)
    return x_values, yield_values, forward_values
