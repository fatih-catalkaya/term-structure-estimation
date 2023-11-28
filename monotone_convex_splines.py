import bisect
import math
from datetime import datetime
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import scipy.integrate as integrate


class InterpolantPiece:
    def __init__(self, max_x: float, function: Callable[[float], float]):
        self.max_x = max_x
        self.function = function

    def __call__(self, *args, **kwargs) -> float:
        return self.function(*args, **kwargs)


class Interpolant:
    def __init__(self):
        self.pieces: list[InterpolantPiece] = []

    def add_piece(self, interpolant_piece: InterpolantPiece):
        self.pieces.append(interpolant_piece)
        self.pieces = sorted(self.pieces, key=lambda piece: piece.max_x)

    def __call__(self, x: float) -> float:
        # Find correct piece
        pieces_max_x = [piece.max_x for piece in self.pieces]
        piece_pos = bisect.bisect_right(pieces_max_x, x)
        piece_pos = min(piece_pos, len(self.pieces) - 1)  # bounds check
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


def make_r1_function(g0: float, g1: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        return g0 * (1.0 - 4.0 * x + 3.0 * math.pow(x, 2)) + g1 * (-2.0 * x + 3.0 * math.pow(x, 2))

    return evaluate


def make_r2_function(g0: float, g1: float, eta: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        if x <= eta:
            return g0
        return g0 + (g1 - g0) * math.pow((x - eta) / (1.0 - eta), 2.0)

    return evaluate


def make_r3_function(g0: float, g1: float, eta: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        if x <= eta:
            return g1 + (g0 - g1) * math.pow((eta - x) / eta, 2)
        return g1

    return evaluate


def make_r4_function(g0: float, g1: float, eta: float, alpha: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        if x <= eta:
            return alpha + (g0 - alpha) * math.pow((eta - x) / eta, 2)
        return alpha + (g1 - alpha) * math.pow((x - eta) / (1.0 - eta), 2)

    return evaluate


def enforce_monotonicity(ytms: list[float], disc_forwards: list[float], inst_forwards: list[float]) -> Interpolant:
    function_g = Interpolant()
    for i in range(len(disc_forwards)):
        disc = disc_forwards[i]
        g0 = inst_forwards[i] - disc
        g1 = inst_forwards[i + 1] - disc
        if check_region_1(g0, g1):
            # Case 1
            interpolant_piece = InterpolantPiece(ytms[i], make_r1_function(g0, g1))
            function_g.add_piece(interpolant_piece)
        elif check_region_2(g0, g1):
            # Case 2
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            function_g.add_piece(InterpolantPiece(ytms[i], make_r2_function(g0, g1, eta)))
        elif check_region_3(g0, g1):
            # Case 3
            eta = 3.0 * (g1 / (g1 - g0))
            function_g.add_piece(InterpolantPiece(ytms[i], make_r3_function(g0, g1, eta)))
        else:
            # Case 4
            eta = g1 / (g1 + g0)
            alpha = -((g0 * g1) / (g0 + g1))
            function_g.add_piece(InterpolantPiece(ytms[i], make_r4_function(g0, g1, eta, alpha)))
    return function_g


def make_forward_function(fcn_g: Callable[[float], float],
                          t0: float,
                          max_x: float,
                          discrete_forward: float) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        return fcn_g((x - t0) / (max_x - t0)) + discrete_forward

    return evaluate


def compute_forward_interpolant(function_g: Interpolant, discrete_forwards: list[float]) -> Interpolant:
    forward_interpolant = Interpolant()
    for i in range(len(function_g.pieces)):
        t0_idx = i - 1
        t0 = 0 if t0_idx < 0 else function_g.pieces[t0_idx].max_x
        piece = function_g.pieces[i]
        f = make_forward_function(piece.function, t0, piece.max_x, discrete_forwards[i])
        forward_interpolant.add_piece(InterpolantPiece(piece.max_x, f))
    return forward_interpolant


def do_stuff(df: pd.DataFrame, today: datetime):
    unique_maturities = df["maturity"].unique()
    yields = []
    for maturity in unique_maturities:
        bonds_for_maturity = df[df["maturity"] == maturity]
        yields.append(np.mean(bonds_for_maturity["yield"]))
    ytms = [(row - today).days / 365.0 for row in unique_maturities]

    #yields = [0.0202, 0.0230, 0.0278, 0.0320, 0.0373]
    #ytms = [1, 2, 3, 4, 5]

    discrete_forwards = compute_discrete_forwards(yields, ytms)
    instantaneous_forwards = compute_instantaneous_forwards(discrete_forwards)
    function_g = enforce_monotonicity(ytms, discrete_forwards, instantaneous_forwards)
    forward_interpolant = compute_forward_interpolant(function_g, discrete_forwards)

    x_values = np.linspace(min(ytms), max(ytms), num=50)
    forward_values = [forward_interpolant(x) for x in x_values]
    yield_values = [(1.0 / x) * integrate.quad(lambda t: forward_interpolant(t), 0, x)[0] for x in x_values]
    plt.plot(x_values, yield_values)
    plt.show()
    print("stuff")
