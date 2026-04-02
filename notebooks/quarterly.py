from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


# =========================================================
# 0) CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

FULL_START = "1990-01-01"
FULL_END = "2025-12-31"
FOCUS_START = "1995-01-01"
FOCUS_END = "2005-12-31"

HAC_LAGS = 1
VAR_MAXLAGS = 4
IRF_PERIODS = 8

GDP_TARGETS = [
    "US GDP_growth",
    "GDP - Personal Consumption_growth",
    "GDP - Goods Consumption_growth",
    "GDP - Durable Goods Consumption_growth",
    "GDP - Non durable Goods_growth",
    "GDP - Service_growth",
    "GDP - Investment_growth",
]

TARGET_LABELS = {
    "US GDP_growth": "US GDP",
    "GDP - Personal Consumption_growth": "Personal Cons.",
    "GDP - Goods Consumption_growth": "Goods Cons.",
    "GDP - Durable Goods Consumption_growth": "Durable Goods",
    "GDP - Non durable Goods_growth": "Non-durable Goods",
    "GDP - Service_growth": "Services",
    "GDP - Investment_growth": "Investment",
}


# =========================================================
# 1) LOAD DATA
# =========================================================
def load_data() -> pd.DataFrame:
    """Load quarterly GDP data and merge with oil returns."""
    quarterly_path = PROJECT_ROOT / "Data" / "processed" / "quarterly_model.csv"
    daily_path = PROJECT_ROOT / "Data" / "processed" / "daily_model.csv"

    for p in [quarterly_path, daily_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    quarterly = pd.read_csv(quarterly_path, parse_dates=["Date"]).sort_values("Date")
    daily = pd.read_csv(daily_path, parse_dates=["Date"]).sort_values("Date")

    daily["Quarter"] = daily["Date"].dt.to_period("Q")

    q_oil = (
        daily.groupby("Quarter", as_index=False)["Brent futures_ret"]
        .sum()
        .rename(columns={"Brent futures_ret": "brent_q_ret"})
    )

    q_oil["Date"] = q_oil["Quarter"].dt.to_timestamp("Q")

    df = quarterly.merge(q_oil[["Date", "brent_q_ret"]], on="Date", how="left")
    return df.reset_index(drop=True)


# =========================================================
# 2) FEATURES
# =========================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["oil_pos"] = df["brent_q_ret"].clip(lower=0)

    pos_vals = df.loc[df["brent_q_ret"] > 0, "brent_q_ret"].dropna()
    threshold = pos_vals.quantile(0.90) if len(pos_vals) > 0 else 0.0

    df["oil_large_shock"] = (df["brent_q_ret"] >= threshold).astype(int)
    df["oil_pos_x_large"] = df["oil_pos"] * df["oil_large_shock"]

    df["oil_pos_lag1"] = df["oil_pos"].shift(1)
    df["oil_pos_lag2"] = df["oil_pos"].shift(2)

    for col in GDP_TARGETS:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_t1"] = df[col].shift(-1)

    return df


# =========================================================
# 3) OLS
# =========================================================
def fit_ols(df, y_col, x_cols):
    tmp = df[[y_col] + x_cols].dropna()
    if tmp.empty:
        return None

    X = sm.add_constant(tmp[x_cols])
    y = tmp[y_col]

    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


# =========================================================
# 4) VAR + GRANGER
# =========================================================
def run_var(df, target):
    tmp = df[["brent_q_ret", target]].dropna()

    model = VAR(tmp)
    lag = model.select_order(VAR_MAXLAGS).bic
    lag = max(1, int(lag))

    fitted = model.fit(lag)

    gc = grangercausalitytests(tmp, maxlag=lag, verbose=False)
    pval = gc[lag][0]["ssr_ftest"][1]

    return lag, pval


# =========================================================
# 5) MAIN
# =========================================================
def main():
    df = load_data()
    df = build_features(df)

    print(df.head())

    for tgt in GDP_TARGETS:
        if tgt not in df.columns:
            continue

        print(f"\n=== {tgt} ===")

        model = fit_ols(df, f"{tgt}_t1", ["oil_pos", f"{tgt}_lag1"])
        if model:
            print(model.summary())

        lag, pval = run_var(df, tgt)
        print(f"VAR lag={lag}, Granger p-value={pval:.4f}")


if __name__ == "__main__":
    main()