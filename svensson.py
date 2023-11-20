import datetime

import numpy as np
import pandas as pd
import math

from scipy import optimize
import utils


def compute_svensson(beta: np.ndarray, tau_1: float, tau_2: float, ytm: float) -> float:
    """
    Berechnet einen Zinssatz auf Grundlage der Svensson-Funktion
    @param beta: Beta-Vektor für Svensson-Kurve
    @param tau_1: Tau_1 Parameter für Svensson-Kurve
    @param tau_2: Tau_2 Parameter für Svensson-Kurve
    @param ytm: Anzahl der Jahre bis zum Laufzeitende
    @return: Zinssatz für angefragte Laufzeit der Anleihe
    """
    assert len(beta) == 4
    frac1 = ((1. - math.exp((-ytm) / tau_1)) / (ytm / tau_1))
    nelson_siegel = float(beta[0] + beta[1] * frac1 + beta[2] * (frac1 - math.exp(-(ytm / tau_1))))
    svensson = float(
        beta[3] * (((1.0 - math.exp((-ytm) / tau_2)) / (ytm / tau_2)) - (math.exp(-(ytm / tau_2)))))
    return nelson_siegel + svensson


def compute_theoretical_bond_price(bond: pd.Series, notional: float, beta: list[float], tau_1: float,
                                   tau_2: float, today: datetime) -> float:
    if bond["coupon"] == 0:
        ytm = (bond["maturity"] - today).days / 365.0
        interest_rate = compute_svensson(np.array(beta), tau_1, tau_2, ytm) / 100.0
        discount_factor = utils.compute_discount_factor(interest_rate, ytm)
        # We know, that the price of a zero-coupon bond = discount factor
        return discount_factor * notional
    else:
        maturity = bond["maturity"]
        dtm = (maturity - today).days
        if dtm < 365:
            # Bond matures in less than one year. In $dtm days we get paid face value + coupon
            payment_notional = notional + notional * (bond["coupon"] / 100.0)
            ytm = dtm / 365.0
            interest_rate = compute_svensson(np.array(beta), tau_1, tau_2, ytm) / 100.0
            discount_factor = utils.compute_discount_factor(interest_rate, ytm)
            return discount_factor * payment_notional
        elif dtm == 365:
            # Bond matures in exactly one year. We have one coupon payment today and one
            # coupon payment + payment of face value in 365 days
            coupon_notional = notional * (bond["coupon"] / 100.0)
            payment_notional = notional + coupon_notional
            ytm = dtm / 365.0
            interest_rate = compute_svensson(np.array(beta), tau_1, tau_2, ytm) / 100.0
            discount_factor = utils.compute_discount_factor(interest_rate, ytm)
            return discount_factor * payment_notional + coupon_notional
        else:
            coupon_notional = notional * (bond["coupon"] / 100.0)
            days_until_next_coupon = dtm % 365
            number_of_pending_coupons = math.floor(dtm / 365)
            # Compute list of years till payment until all coupons are paid
            ytms = [years_till_payment + (days_until_next_coupon / 365.0) for years_till_payment in
                    range(0, number_of_pending_coupons + 1)]
            bond_price = 0.0
            for ytm in ytms:
                interest_rate = compute_svensson(np.array(beta), tau_1, tau_2, ytm) / 100.0
                discount_factor = utils.compute_discount_factor(interest_rate, ytm)
                bond_price += discount_factor * coupon_notional
            # Discount face value and add to bond price
            bond_ytm = (bond["maturity"] - today).days / 365.0
            interest_rate = compute_svensson(np.array(beta), tau_1, tau_2, bond_ytm) / 100.0
            discount_factor = utils.compute_discount_factor(interest_rate, bond_ytm)
            bond_price += discount_factor * notional
            return bond_price


def error_function(x, df: pd.DataFrame) -> list[float]:
    today = datetime.datetime(year=2023, month=11, day=10)
    residuals = []
    beta = x[0:4]
    tau_1 = x[4]
    tau_2 = x[5]
    for idx, row in df.iterrows():
        theoretical_price = compute_theoretical_bond_price(row, 100, beta, tau_1, tau_2, today)
        theoretical_bond_yield = utils.compute_yield_to_maturity(row, theoretical_price, 100, today)
        bond_yield = row["yield"] / 100.0
        residuals.append(theoretical_bond_yield - bond_yield)
    return residuals


def compute_parameters(df: pd.DataFrame) -> tuple[list[float], float, float, optimize.OptimizeResult]:
    param = np.zeros(shape=6)
    param[0] = utils.compute_beta_0(df)
    param[1] = utils.compute_beta_1(df, float(param[0]))
    param[2] = -1.0
    param[3] = -1.0
    param[4] = 1.0
    param[5] = 1.0
    lower_bounds = [(utils.compute_yield_of_longest_bond(df) - 3), -30, -30, -30, 10e-4, 10e-4]
    upper_bounds = [(utils.compute_yield_of_longest_bond(df) + 3), 30, 30, 30, 30, 30]
    res = optimize.least_squares(error_function, param, kwargs={"df": df},
                                 bounds=(lower_bounds, upper_bounds),
                                 ftol=10e-12, xtol=10e-12, gtol=10e-12)
    res_vec = res["x"]
    beta = res_vec[0:4]
    tau_1 = res_vec[4]
    tau_2 = res_vec[5]
    return beta, tau_1, tau_2, res
