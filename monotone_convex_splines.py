from datetime import datetime

import numpy as np
import pandas as pd


def compute_discrete_forwards(yields: list[float], ytms: list[float]) -> list[float]:
    discrete_forwards = [yields[0]]
    for i in range(1, len(ytms)):
        discrete_forward = (yields[i] * ytms[i] - yields[i - 1] * ytms[i - 1]) / (ytms[i] - ytms[i - 1])
        discrete_forwards.append(discrete_forward)
    return discrete_forwards


def compute_instantaneous_forwards(discrete_forwards: list[float]) -> list[float]:
    inst_forwards: list[float] = []
    for i in range(0, len(discrete_forwards) - 1):
        inst_forwards.append(0.5 * (discrete_forwards[i] + discrete_forwards[i + 1]))
    inst_forwards = [discrete_forwards[0] - 0.5 * (inst_forwards[0] - discrete_forwards[0])] + inst_forwards
    inst_forwards.append(discrete_forwards[len(discrete_forwards) - 1] - 0.5 * (
            inst_forwards[len(inst_forwards) - 1] - discrete_forwards[len(discrete_forwards) - 1]))
    return inst_forwards


def do_stuff(df: pd.DataFrame, today: datetime):
    # unique_maturities = df["maturity"].unique()
    # yields = []
    # for maturity in unique_maturities:
    #     bonds_for_maturity = df[df["maturity"] == maturity]
    #     yields.append(np.mean(bonds_for_maturity["yield"]))
    # ytms = [(row - today).days / 365.0 for row in unique_maturities]

    yields = [0.0202, 0.0230, 0.0278, 0.0320, 0.0373]
    ytms = [1, 2, 3, 4, 5]

    discrete_forwards = compute_discrete_forwards(yields, ytms)
    instantaneous_forwards = compute_instantaneous_forwards(discrete_forwards)
    print(instantaneous_forwards)
