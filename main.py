from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

import data_importer
import monotone_convex_splines
import monotone_cubic_splines
import nelson_siegel
import svensson


def plot_nelson_siegel_and_data(df: pd.DataFrame, beta: list[float], tau1: float) -> None:
    today = datetime(year=2023, month=11, day=10)
    published_params = ([0.94381, 2.59860, 3.58184, 6.28684], 0.61208, 13.74114)
    dtm = df["maturity"]
    dtm = dtm.apply(lambda row: (row - today).days / 365.0)
    yields = df["yield"]
    ns_yields = [nelson_siegel.compute_nelson_siegel(np.array(beta), tau1, maturity) for maturity in dtm]
    real_yields = [
        svensson.compute_svensson(np.array(published_params[0]), published_params[1], published_params[2], maturity) for
        maturity in
        dtm]
    plt.plot(dtm, yields, marker="x", label="German bonds")
    plt.plot(dtm, ns_yields, label="Estimated Nelson-Siegel curve")
    plt.plot(dtm, real_yields, label="Published Svensson curve")
    plt.ylabel("Yield")
    plt.xlabel("Years to maturity")
    plt.title("Nelson-Siegel curve")
    plt.legend(loc="upper right")
    fig = plt.gcf()
    fig.savefig("nelson-siegel.png", dpi=300)
    plt.show()


def plot_svensson_and_data(df: pd.DataFrame, beta: list[float], tau1: float, tau2: float) -> None:
    today = datetime(year=2023, month=11, day=10)
    published_params = ([0.94381, 2.59860, 3.58184, 6.28684], 0.61208, 13.74114)
    dtm = df["maturity"]
    dtm = dtm.apply(lambda row: (row - today).days / 365.0)
    yields = df["yield"]
    sv_yields = [svensson.compute_svensson(np.array(beta), tau1, tau2, maturity) for maturity in dtm]
    real_yields = [
        svensson.compute_svensson(np.array(published_params[0]), published_params[1], published_params[2], maturity) for
        maturity in
        dtm]
    plt.plot(dtm, yields, marker="x", label="German bonds")
    plt.plot(dtm, sv_yields, label="Estimated Svensson curve")
    plt.plot(dtm, real_yields, label="Published Svensson curve")
    plt.ylabel("Yield")
    plt.xlabel("Years to maturity")
    plt.title("Svensson curve")
    plt.legend(loc="upper right")
    fig = plt.gcf()
    fig.savefig("svensson.png", dpi=300)
    plt.show()


def plot_mcc(x_values: list[float], yield_values: list[float], forward_values: list[float]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle("Computed monotone cubic splines")
    ax0.plot(x_values, yield_values)
    ax0.set_title("Yield curve")
    ax1.plot(x_values, forward_values)
    ax1.set_title("Instantaneous forward rate")
    fig.savefig("monotone-cubic.png", dpi=300)
    fig.show()


def plot_mcx(x_values: list[float], yield_values: list[float], forward_values: list[float]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle("Computed monotone convex splines")
    ax0.plot(x_values, yield_values)
    ax0.set_title("Yield curve")
    ax1.plot(x_values, forward_values)
    ax1.set_title("Instantaneous forward rate")
    fig.savefig("monotone-convex.png", dpi=300)
    fig.show()


if __name__ == "__main__":
    # Import data, provided from Deutsche Bundesbank
    df = data_importer.import_data()

    # Fix date to today's date (as of writing)
    today = datetime(year=2023, day=10, month=11)

    # Published Data for 11th November 2023
    published_beta = [0.94381, 2.59860, 3.58184, 6.28684]
    published_tau1 = 0.61208
    published_tau2 = 13.74114

    # Compute and plot Nelson-Siegel-Parameter
    ns_beta, ns_tau_1, _ = nelson_siegel.compute_parameters(df)
    plot_nelson_siegel_and_data(df, ns_beta, ns_tau_1)

    # Compute and plot Svensson-Parameter
    sv_beta, sv_tau_1, sv_tau_2, _ = svensson.compute_parameters(df)
    plot_svensson_and_data(df, sv_beta, sv_tau_1, sv_tau_2)

    # Compute Monotone Cubic splines for yield curve and instantaneous forward rates
    mcc_x_values, mcc_yields, mcc_forwards = monotone_cubic_splines.compute_splines(df, today)
    plot_mcc(mcc_x_values, mcc_yields, mcc_forwards)

    # Compute Monotone Convex splines for yield curve and instantaneous forward rates
    mcx_x_values, mcx_yields, mcx_forwards = monotone_convex_splines.compute_splines(df, today)
    plot_mcx(mcx_x_values, mcx_yields, mcx_forwards)

    # Print table with Nelson-Siegel and Svensson estimates and published values
    table = PrettyTable(["Parameter", "Computed value (Nelson-Siegel)", "Computed value (Svensson)", "Published value"])
    table.add_rows([
        ["Beta0", ns_beta[0], sv_beta[0], published_beta[0]],
        ["Beta1", ns_beta[1], sv_beta[1], published_beta[1]],
        ["Beta2", ns_beta[2], sv_beta[2], published_beta[2]],
        ["Beta3", "---", sv_beta[3], published_beta[3]],
        ["Tau 1", ns_tau_1, sv_tau_1, published_tau1],
        ["Tau 2", "---", sv_tau_2, published_tau2],
    ])
    table.align["Parameter"] = "l"
    table.align["Computed value (Nelson-Siegel)"] = "r"
    table.align["Computed value (Svensson)"] = "r"
    table.align["Published value"] = "r"
    table.float_format = "0.8"
    print(table)
