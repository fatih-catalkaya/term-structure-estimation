import bisect
import math
from datetime import datetime
from typing import Callable, Any

import numpy as np
import pandas as pd


class InterpolantPiece:
    def __init__(self, min_x: float, function: Callable[[float], float]):
        self.min_x = min_x
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Interpolant:
    def __init__(self):
        self.pieces: list[InterpolantPiece] = []

    def add_piece(self, interpolant_piece: InterpolantPiece):
        self.pieces.append(interpolant_piece)
        self.pieces = sorted(self.pieces, key=lambda piece: piece.min_x)

    def __call__(self, x: float):
        # Find correct piece
        piece_pos = bisect.bisect_right(self.pieces, x)
        piece_pos = min(0, piece_pos - 1)  # bounds check
        return self.pieces[piece_pos](x)


def check_region_1(g0: float, g1: float) -> bool:
    return (g0 < 0 and -0.5 * g0 <= g1 and g1 <= -2.0 * g0) or (g0 < 0 and -0.5 * g0 >= g1 and g1 >= -2.0 * g0)


def check_region_2(g0: float, g1: float) -> bool:
    return (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0)


def check_region_3(g0: float, g1: float) -> bool:
    return (g0 > 0 and 0 > g1 and g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 and g1 < -0.5 * g0)


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


def enforce_monotonicity(ytms: list[float], disc_forwards: list[float], inst_forwards: list[float]):
    function_g = Interpolant()
    for i in range(len(disc_forwards)):
        disc = disc_forwards[i]
        g0 = inst_forwards[i] - disc
        g1 = inst_forwards[i + 1] - disc
        if check_region_1(g0, g1):
            # Case 1
            g = lambda x: g0 * (1.0 - 4.0 * x + 3.0 * (x ** 2.0)) + g1 * (-2.0 * x + 3.0 * (x ** 2.0))
            interpolant_piece = InterpolantPiece(ytms[i], g)
            function_g.add_piece(interpolant_piece)
        elif check_region_2(g0, g1):
            # Case 2
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            ytm_first_part = ytms[i] + eta
            ytm_second_part = ytms[i + 1]
            g_first_part = lambda x: g0
            g_second_part = lambda x: g0 + (g1 - g0) * math.pow((x - eta) / (1.0 - eta), 2.0)
            function_g.add_piece(InterpolantPiece(ytm_first_part, g_first_part))
            function_g.add_piece(InterpolantPiece(ytm_second_part, g_second_part))
        elif check_region_3(g0, g1):
            # Case 3
            eta = 3.0*(g1/(g1-g0))
            ytm_first_part = ytms[i] + eta
            ytm_second_part = ytms[i + 1]
            g_first_part = lambda x: g1+(g0-g1)*math.pow((eta-x)/eta, 2)
            g_second_part = lambda x: g1
            function_g.add_piece(InterpolantPiece(ytm_first_part, g_first_part))
            function_g.add_piece(InterpolantPiece(ytm_second_part, g_second_part))
        else:
            # Case 4
            continue


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
    enforce_monotonicity(discrete_forwards, instantaneous_forwards)
    print(instantaneous_forwards)
