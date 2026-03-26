import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA


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
df["brent_pos_ret"] = df["Brent futures_ret"].clip(lower=0)

positive_brent = df.loc[df["Brent futures_ret"] > 0, "Brent futures_ret"].dropna()
if len(positive_brent) == 0:
    raise ValueError("No positive Brent returns found in the sample.")

threshold_90 = positive_brent.quantile(0.90)

df["oil_large_pos_shock"] = (df["Brent futures_ret"] >= threshold_90).astype(int)
df["oil_pos_shock"] = (df["Brent futures_ret"] > 0).astype(int)

df["brent_pos_ma3"] = df["brent_pos_ret"].rolling(3).mean()
df["brent_pos_ma5"] = df["brent_pos_ret"].rolling(5).mean()
df["brent_pos_lag1"] = df["brent_pos_ret"].shift(1)
df["brent_pos_lag2"] = df["brent_pos_ret"].shift(2)
df["brent_pos_x_large"] = df["brent_pos_ret"] * df["oil_large_pos_shock"]


# =========================================================
# Volatility proxies
# =========================================================
df["spx_vol"] = df["S&P500_ret"].abs()
df["msci_vol"] = df["MSCI World_ret"].abs()
df["us10y_vol"] = df["US 10-year Rate_chg"].abs()
df["us2y_vol"] = df["US 2-year Rate_chg"].abs()
df["hy_vol"] = df["High yield index_chg"].abs()
df["gold_vol"] = df["Gold_ret"].abs()


# =========================================================
# Helpers
# =========================================================
def safe_name(x):
    return (
        x.replace(" ", "_")
         .replace("-", "")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
    )


def pseudo_r2(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst == 0:
        return np.nan
    return 1 - sse / sst


def fit_armax(data, y_var, exog_vars, p, q):
    """
    ARMA(p,q) with exogenous regressors = ARMAX
    Implemented as ARIMA(p,0,q) with exog.
    """
    use_cols = [y_var] + exog_vars
    tmp = data[use_cols].dropna().copy()

    y = tmp[y_var]
    X = tmp[exog_vars] if len(exog_vars) > 0 else None

    model = ARIMA(
        endog=y,
        exog=X,
        order=(p, 0, q),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()

    fitted = model.fittedvalues
    pr2 = pseudo_r2(y.loc[fitted.index], fitted)

    return model, len(tmp), pr2


def compare_armax_models(df, target_name, specs, p_values=(1, 2), q_values=(0, 1, 2)):
    rows = []
    fitted_models = {}

    for spec_name, regressors in specs.items():
        for p in p_values:
            for q in q_values:
                model_name = f"{spec_name}_arma({p},{q})"
                try:
                    model, nobs, pr2 = fit_armax(df, target_name, regressors, p, q)
                    fitted_models[model_name] = model

                    rows.append({
                        "dependent_variable": target_name,
                        "model_name": model_name,
                        "spec_name": spec_name,
                        "p": p,
                        "q": q,
                        "n_obs": nobs,
                        "pseudo_r_squared": pr2,
                        "aic": model.aic,
                        "bic": model.bic,
                        "n_regressors": len(regressors),
                        "regressors": " | ".join(regressors),
                    })
                except Exception as e:
                    rows.append({
                        "dependent_variable": target_name,
                        "model_name": model_name,
                        "spec_name": spec_name,
                        "p": p,
                        "q": q,
                        "n_obs": np.nan,
                        "pseudo_r_squared": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_regressors": len(regressors),
                        "regressors": " | ".join(regressors),
                        "error": str(e),
                    })

    comparison = pd.DataFrame(rows).sort_values(
        by=["pseudo_r_squared", "aic"],
        ascending=[False, True],
        na_position="last"
    ).reset_index(drop=True)

    best_model_name = comparison.iloc[0]["model_name"]
    best_model = fitted_models.get(best_model_name)

    return comparison, best_model_name, best_model


def export_results(target_name, comparison, best_name, best_model):
    out_name = safe_name(target_name)

    comparison.to_csv(OUTPUT_DIR / f"{out_name}_arma_comparison.csv", index=False)

    if best_model is not None:
        try:
            params = best_model.params
            bse = best_model.bse
            pvalues = best_model.pvalues

            coef_table = pd.DataFrame({
                "variable": params.index,
                "coef": params.values,
                "std_err": bse.values,
                "p_value": pvalues.values,
            })
            coef_table.to_csv(OUTPUT_DIR / f"{out_name}_best_arma_coefficients.csv", index=False)
        except Exception:
            pass

        with open(OUTPUT_DIR / f"{out_name}_best_arma_summary.txt", "w") as f:
            f.write(best_model.summary().as_text())


# =========================================================
# Candidate explanatory specs
# Same spirit as your current code, but now inside ARMAX
# =========================================================

spx_specs = {
    "spx_baseline": [
        "brent_pos_ret",
        "oil_large_pos_shock",
    ],
    "spx_smoothed": [
        "brent_pos_ma3",
        "oil_large_pos_shock",
    ],
    "spx_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
    ],
    "spx_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "msci_vol",
        "hy_vol",
        "us10y_vol",
    ],
}

us10y_specs = {
    "us10y_baseline": [
        "brent_pos_ret",
        "oil_large_pos_shock",
    ],
    "us10y_smoothed": [
        "brent_pos_ma3",
        "oil_large_pos_shock",
    ],
    "us10y_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
    ],
    "us10y_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "us2y_vol",
        "spx_vol",
        "hy_vol",
    ],
}

hy_specs = {
    "hy_baseline": [
        "brent_pos_ret",
        "oil_large_pos_shock",
    ],
    "hy_smoothed": [
        "brent_pos_ma3",
        "oil_large_pos_shock",
    ],
    "hy_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
    ],
    "hy_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "spx_vol",
        "us10y_vol",
        "gold_vol",
    ],
}

gold_specs = {
    "gold_baseline": [
        "brent_pos_ret",
        "oil_large_pos_shock",
    ],
    "gold_smoothed": [
        "brent_pos_ma3",
        "oil_large_pos_shock",
    ],
    "gold_interaction": [
        "brent_pos_ret",
        "oil_large_pos_shock",
        "brent_pos_x_large",
    ],
    "gold_cross_asset": [
        "brent_pos_ret",
        "oil_large_pos_shock",
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
    ("spx_vol", spx_specs),
    ("us10y_vol", us10y_specs),
    ("hy_vol", hy_specs),
    ("gold_vol", gold_specs),
]

for target_name, specs in targets:
    print("\n" + "=" * 100)
    print(f"TARGET: {target_name}")
    print("=" * 100)

    comparison, best_name, best_model = compare_armax_models(
        df,
        target_name,
        specs,
        p_values=(1, 2),
        q_values=(0, 1, 2),
    )

    export_results(target_name, comparison, best_name, best_model)

    print(comparison[[
        "model_name", "pseudo_r_squared", "aic", "bic", "n_obs"
    ]].head(10))

    print(f"\nBest ARMA model for {target_name}: {best_name}")
    if best_model is not None:
        print(best_model.summary())

        top_row = comparison.iloc[0]
        summary_rows.append({
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_pseudo_r_squared": top_row["pseudo_r_squared"],
            "aic": top_row["aic"],
            "bic": top_row["bic"],
            "n_obs": int(top_row["n_obs"]),
        })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by="best_pseudo_r_squared", ascending=False
).reset_index(drop=True)

summary_df.to_csv(OUTPUT_DIR / "best_arma_volatility_models_summary.csv", index=False)

print("\n" + "=" * 100)
print("BEST ARMA VOLATILITY MODELS SUMMARY")
print("=" * 100)
print(summary_df)