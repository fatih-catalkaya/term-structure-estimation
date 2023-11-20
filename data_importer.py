import pandas as pd


def import_data() -> pd.DataFrame:
    # Read CSV
    df = pd.read_csv("german_bond_data.csv", sep=";", header=None, decimal=",")

    # Compute ISIN, since it is given as three seperate columns
    df["isin"] = df[df.columns[0]].astype(str) + df[df.columns[1]].astype(str) + df[df.columns[2]].astype(str)

    # Rename columns for easier semantics
    column_rename_mapping = {
        df.columns[3]: "coupon",
        df.columns[4]: "description",
        df.columns[5]: "maturity",
        df.columns[6]: "residual_life_ym",
        df.columns[7]: "issue_vol_bn",
        df.columns[8]: "clean_price",
        df.columns[9]: "yield",
        df.columns[10]: "dirty_price",
    }
    df = df.rename(columns=column_rename_mapping)

    # Drop unnecessary columns
    df = df.drop(columns=[df.columns[0], df.columns[1], df.columns[2]])

    # Drop Inflation Index Bonds
    df = df[~df["description"].str.contains("index")]

    # Drop other unnecessary columns
    df = df.drop(columns=["residual_life_ym", "issue_vol_bn"])

    # Parse column maturity to datetime objects
    df["maturity"] = pd.to_datetime(df["maturity"], format="%d.%m.%Y")
    return df
