import math
from datetime import datetime

import pandas as pd
from scipy import optimize


def compute_beta_0(df: pd.DataFrame) -> float:
    # The paper by Schich initializes beta_0 with the mean yield-to-maturity
    # of the three bonds with the longest time to maturity.
    df_ordered = df.sort_values(by=["maturity"], ascending=False)
    mean_ytm = df_ordered[0:3]["yield"].mean()
    return mean_ytm / 100.0


def compute_beta_1(df: pd.DataFrame, beta_0: float) -> float:
    # The paper by Schich initializes beta_1 as follows:
    # let x denote the yield-to-maturity
    # of the bond with the shortest time to maturity.
    # Then beta_1 + beta_0 := x <=> beta_1 = x - beta_0
    df_ordered = df.sort_values(by=["maturity"], ascending=True)
    return (df_ordered[:1]["yield"].values[0] / 100.0) - beta_0


def compute_yield_of_longest_bond(df: pd.DataFrame) -> float:
    return df.sort_values(by=["maturity"], ascending=False)[:1]["yield"].values[0] / 100.0


def compute_discount_factor(interest_rate: float, ytm: float) -> float:
    return 1.0 / math.pow(1.0 + interest_rate, ytm)


def error_yield_computation(x: float, bond: pd.Series, bond_price: float, notional: float, today: datetime) -> float:
    dtm = (bond["maturity"] - today).days
    days_until_next_coupon = dtm % 365.0
    number_of_pending_coupons = math.floor(dtm / 365)
    coupon_notional = notional * (bond["coupon"] / 100.0)
    ytms = [years_till_payment + (days_until_next_coupon / 365.0) for years_till_payment in
            range(0, number_of_pending_coupons + 1)]
    bond_price_yield = 0.0
    for ytm in ytms:
        bond_price_yield += coupon_notional / math.pow(1.0 + x, ytm)
    # Discount face value and add to bond price
    bond_ytm = (bond["maturity"] - today).days / 365.0
    bond_price_yield += notional / math.pow(1.0 + x, bond_ytm)

    return bond_price - bond_price_yield


def compute_yield_to_maturity(bond: pd.Series, bond_price: float, notional: float, today: datetime) -> float:
    if bond["coupon"] == 0:
        # For zero-coupon bonds we can derive its yield analytically
        ytm = (bond["maturity"] - today).days / 365.0
        bond_yield = math.pow(notional / bond_price, 1. / ytm) - 1.0
        return bond_yield
    else:
        # For coupon bonds we have to perform a least-squares optimization
        res = optimize.least_squares(error_yield_computation, 0.05,
                                     kwargs={"bond": bond, "bond_price": bond_price,
                                             "notional": notional, "today": today})
        return res["x"][0]
