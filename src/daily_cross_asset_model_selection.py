# src/best_positive_brent_explanatory_model.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Data" / "processed" / "daily_model.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")


# =========================================================
# Load data
# =========================================================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Loaded file:", DATA_PATH)
print("Shape:", df.shape)


# =========================================================
# Required columns
# =========================================================
required_cols = [
    "Date",
    "Brent futures_ret",
    "S&P500_ret",
    "MSCI World_ret",
    "US 10-year Rate_chg",
    "US 2-year Rate_chg",
    "High yield index_chg",
    "Gold_ret",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")


# =========================================================
# Positive Brent explanatory variables
# =========================================================
# Positive Brent move only
df["brent_pos_ret"] = df["Brent futures_ret"].clip(lower=0)

# Large positive shock dummy = top decile of positive Brent moves
positive_brent = df.loc[df["Brent futures_ret"] > 0, "Brent futures_ret"].dropna()
if len(positive_brent) == 0:
    raise ValueError("No positive Brent returns found in the sample.")

threshold_90 = positive_brent.quantile(0.90)
df["oil_large_pos_shock"] = (df["Brent futures_ret"] >= threshold_90).astype(int)

# Positive shock dummy
df["oil_pos_shock"] = (df["Brent futures_ret"] > 0).astype(int)

# Smoothers to reduce daily noise
df["brent_pos_ma3"] = df["brent_pos_ret"].rolling(3).mean()
df["brent_pos_ma5"] = df["brent_pos_ret"].rolling(5).mean()

# Lags
df["brent_pos_lag1"] = df["brent_pos_ret"].shift(1)
df["brent_pos_lag2"] = df["brent_pos_ret"].shift(2)

# Interaction term
df["brent_pos_x_large"] = df["brent_pos_ret"] * df["oil_large_pos_shock"]


# =========================================================
# Volatility proxies
# Based on the course: squared and absolute returns are noisy but
# standard daily proxies for conditional volatility.
# =========================================================
df["spx_vol"] = df["S&P500_ret"].abs()
df["msci_vol"] = df["MSCI World_ret"].abs()
df["us10y_vol"] = df["US 10-year Rate_chg"].abs()
df["us2y_vol"] = df["US 2-year Rate_chg"].abs()
df["hy_vol"] = df["High yield index_chg"].abs()
df["gold_vol"] = df["Gold_ret"].abs()

# next-day targets
df["spx_vol_t1"] = df["spx_vol"].shift(-1)
df["us10y_vol_t1"] = df["us10y_vol"].shift(-1)
df["hy_vol_t1"] = df["hy_vol"].shift(-1)
df["gold_vol_t1"] = df["gold_vol"].shift(-1)

# lags of own volatility
df["spx_vol_lag1"] = df["spx_vol"].shift(1)
df["us10y_vol_lag1"] = df["us10y_vol"].shift(1)
df["hy_vol_lag1"] = df["hy_vol"].shift(1)
df["gold_vol_lag1"] = df["gold_vol"].shift(1)


# =========================================================
# Helpers
# =========================================================
def fit_ols_hac(data, y_var, x_vars, hac_lags=5):
    tmp = data[[y_var] + x_vars].dropna().copy()
    X = sm.add_constant(tmp[x_vars])
    y = tmp[y_var]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model, len(tmp)


def safe_name(x):
    return (
        x.replace(" ", "_")
         .replace("-", "")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
    )


def compare_models(df, target_name, specs, hac_lags=5):
    rows = []
    fitted = {}

    for model_name, regressors in specs.items():
        try:
            model, nobs = fit_ols_hac(df, target_name, regressors, hac_lags=hac_lags)
            fitted[model_name] = model

            rows.append({
                "dependent_variable": target_name,
                "model_name": model_name,
                "n_obs": nobs,
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "n_regressors": len(regressors),
                "regressors": " | ".join(regressors),
            })
        except Exception as e:
            rows.append({
                "dependent_variable": target_name,
                "model_name": model_name,
                "n_obs": np.nan,
                "r_squared": np.nan,
                "adj_r_squared": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "n_regressors": len(regressors),
                "regressors": " | ".join(regressors),
                "error": str(e),
            })

    comparison = pd.DataFrame(rows).sort_values(
        by=["r_squared", "adj_r_squared"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    best_model_name = comparison.iloc[0]["model_name"]
    best_model = fitted.get(best_model_name)

    return comparison, best_model_name, best_model


def export_results(target_name, comparison, best_name, best_model):
    out_name = safe_name(target_name)

    comparison.to_csv(OUTPUT_DIR / f"{out_name}_comparison.csv", index=False)

    if best_model is not None:
        coef_table = pd.DataFrame({
            "variable": best_model.params.index,
            "coef": best_model.params.values,
            "std_err": best_model.bse.values,
            "t_stat": best_model.tvalues.values,
            "p_value": best_model.pvalues.values,
        })
        coef_table.to_csv(OUTPUT_DIR / f"{out_name}_best_coefficients.csv", index=False)

        with open(OUTPUT_DIR / f"{out_name}_best_summary.txt", "w") as f:
            f.write(best_model.summary().as_text())

        pd.DataFrame([{
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_r_squared": best_model.rsquared,
            "best_adj_r_squared": best_model.rsquared_adj,
            "aic": best_model.aic,
            "bic": best_model.bic,
            "n_obs": int(best_model.nobs),
        }]).to_csv(OUTPUT_DIR / f"{out_name}_best_metrics.csv", index=False)


# =========================================================
# Candidate explanatory models
# Keep only explanatory variables that make economic sense
# =========================================================

# S&P500 daily volatility
spx_specs = {
    "spx_m1_baseline": [
        "brent_pos_ret",
        "spx_vol_lag1",
    ],
    "spx_m2_large_shock": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "spx_vol_lag1",
    ],
    "spx_m3_smoothed": [
        "brent_pos_ma3",
        "spx_vol_lag1",
    ],
    "spx_m4_smoothed_large": [
        "brent_pos_ma5",
        "oil_large_pos_shock",
        "spx_vol_lag1",
    ],
    "spx_m5_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
        "spx_vol_lag1",
    ],
    "spx_m6_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "spx_vol_lag1",
        "msci_vol",
        "hy_vol",
        "us10y_vol",
    ],
}

# US 10Y daily volatility
us10y_specs = {
    "us10y_m1_baseline": [
        "brent_pos_ret",
        "us10y_vol_lag1",
    ],
    "us10y_m2_large_shock": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "us10y_vol_lag1",
    ],
    "us10y_m3_smoothed": [
        "brent_pos_ma3",
        "us10y_vol_lag1",
    ],
    "us10y_m4_smoothed_large": [
        "brent_pos_ma5",
        "oil_large_pos_shock",
        "us10y_vol_lag1",
    ],
    "us10y_m5_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
        "us10y_vol_lag1",
    ],
    "us10y_m6_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "us10y_vol_lag1",
        "us2y_vol",
        "spx_vol",
        "hy_vol",
    ],
}

# High Yield daily volatility
hy_specs = {
    "hy_m1_baseline": [
        "brent_pos_ret",
        "hy_vol_lag1",
    ],
    "hy_m2_large_shock": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "hy_vol_lag1",
    ],
    "hy_m3_smoothed": [
        "brent_pos_ma3",
        "hy_vol_lag1",
    ],
    "hy_m4_smoothed_large": [
        "brent_pos_ma5",
        "oil_large_pos_shock",
        "hy_vol_lag1",
    ],
    "hy_m5_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
        "hy_vol_lag1",
    ],
    "hy_m6_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "hy_vol_lag1",
        "spx_vol",
        "us10y_vol",
        "gold_vol",
    ],
}

# Gold daily volatility
gold_specs = {
    "gold_m1_baseline": [
        "brent_pos_ret",
        "gold_vol_lag1",
    ],
    "gold_m2_large_shock": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "gold_vol_lag1",
    ],
    "gold_m3_smoothed": [
        "brent_pos_ma3",
        "gold_vol_lag1",
    ],
    "gold_m4_smoothed_large": [
        "brent_pos_ma5",
        "oil_large_pos_shock",
        "gold_vol_lag1",
    ],
    "gold_m5_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
        "gold_vol_lag1",
    ],
    "gold_m6_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "gold_vol_lag1",
        "spx_vol",
        "us10y_vol",
        "msci_vol",
    ],
}


# =========================================================
# Run model selection
# =========================================================
summary_rows = []

targets = [
    ("spx_vol_t1", spx_specs),
    ("us10y_vol_t1", us10y_specs),
    ("hy_vol_t1", hy_specs),
    ("gold_vol_t1", gold_specs),
]

for target_name, specs in targets:
    print("\n" + "=" * 100)
    print(f"TARGET: {target_name}")
    print("=" * 100)

    comparison, best_name, best_model = compare_models(df, target_name, specs, hac_lags=5)
    export_results(target_name, comparison, best_name, best_model)

    print(comparison[["model_name", "n_obs", "r_squared", "adj_r_squared", "aic", "bic"]])

    print(f"\nBest model for {target_name}: {best_name}")
    if best_model is not None:
        print(best_model.summary())

        summary_rows.append({
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_r_squared": best_model.rsquared,
            "best_adj_r_squared": best_model.rsquared_adj,
            "aic": best_model.aic,
            "bic": best_model.bic,
            "n_obs": int(best_model.nobs),
        })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by="best_r_squared", ascending=False
).reset_index(drop=True)

summary_df.to_csv(OUTPUT_DIR / "best_positive_brent_daily_financial_models.csv", index=False)

print("\n" + "=" * 100)
print("BEST MODELS SUMMARY")
print("=" * 100)
print(summary_df)