import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

daily_path = PROCESSED_DIR / "daily_model.csv"
monthly_path = PROCESSED_DIR / "monthly_model.csv"
quarterly_path = PROCESSED_DIR / "quarterly_model.csv"

for p in [daily_path, monthly_path, quarterly_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


# =========================================================
# Load datasets
# =========================================================
daily = pd.read_csv(daily_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
monthly = pd.read_csv(monthly_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
quarterly = pd.read_csv(quarterly_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

print("Daily shape:", daily.shape)
print("Monthly shape:", monthly.shape)
print("Quarterly shape:", quarterly.shape)


# =========================================================
# Build monthly / quarterly oil series from daily Brent log returns
# Sum of daily log returns within period
# =========================================================
daily_oil = daily[["Date", "Brent futures_ret"]].copy()

daily_oil["YearMonth"] = daily_oil["Date"].dt.to_period("M")
monthly_oil = daily_oil.groupby("YearMonth", as_index=False)["Brent futures_ret"].sum()
monthly_oil["Date"] = monthly_oil["YearMonth"].dt.to_timestamp("M")
monthly_oil = monthly_oil.rename(columns={"Brent futures_ret": "Brent_monthly_ret"})[["Date", "Brent_monthly_ret"]]

daily_oil["Quarter"] = daily_oil["Date"].dt.to_period("Q")
quarterly_oil = daily_oil.groupby("Quarter", as_index=False)["Brent futures_ret"].sum()
quarterly_oil["Date"] = quarterly_oil["Quarter"].dt.to_timestamp("Q")
quarterly_oil = quarterly_oil.rename(columns={"Brent futures_ret": "Brent_quarterly_ret"})[["Date", "Brent_quarterly_ret"]]


# =========================================================
# Merge oil into macro datasets
# =========================================================
monthly_reg = monthly.merge(monthly_oil, on="Date", how="left")
quarterly_reg = quarterly.merge(quarterly_oil, on="Date", how="left")


# =========================================================
# Helpers
# =========================================================
def safe_name(x: str) -> str:
    return (
        x.replace(" ", "_")
         .replace("-", "")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
    )


def add_lags(df: pd.DataFrame, cols: list[str], lags=(1, 2)) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def build_positive_oil_features(df: pd.DataFrame, oil_col: str, prefix: str) -> pd.DataFrame:
    df[f"{prefix}_ret_pos"] = df[oil_col].clip(lower=0)

    pos = df.loc[df[oil_col] > 0, oil_col].dropna()
    threshold = pos.quantile(0.90) if len(pos) > 0 else 0.0

    df[f"{prefix}_large_pos_shock"] = (df[oil_col] >= threshold).astype(int)
    df[f"{prefix}_ret_pos_sq"] = df[f"{prefix}_ret_pos"] ** 2
    df[f"{prefix}_ret_pos_x_large"] = df[f"{prefix}_ret_pos"] * df[f"{prefix}_large_pos_shock"]

    # add lags of oil
    df = add_lags(df, [f"{prefix}_ret_pos"], lags=(1, 2))
    return df


def create_t1_targets(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[f"{col}_t1"] = df[col].shift(-1)
    return df


def fit_ols_hac(data: pd.DataFrame, y_var: str, x_vars: list[str], hac_lags: int):
    tmp = data[[y_var] + x_vars].dropna().copy()
    X = sm.add_constant(tmp[x_vars])
    y = tmp[y_var]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model, len(tmp)


def compare_models(df: pd.DataFrame, y_var: str, specs: dict, hac_lags: int):
    rows = []
    fitted = {}

    for model_name, x_vars in specs.items():
        try:
            model, nobs = fit_ols_hac(df, y_var, x_vars, hac_lags)
            fitted[model_name] = model

            rows.append({
                "dependent_variable": y_var,
                "model_name": model_name,
                "n_obs": nobs,
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "regressors": " | ".join(x_vars),
            })
        except Exception as e:
            rows.append({
                "dependent_variable": y_var,
                "model_name": model_name,
                "n_obs": np.nan,
                "r_squared": np.nan,
                "adj_r_squared": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "regressors": " | ".join(x_vars),
                "error": str(e),
            })

    comparison = pd.DataFrame(rows).sort_values(
        by=["r_squared", "adj_r_squared"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    best_name = comparison.iloc[0]["model_name"]
    best_model = fitted.get(best_name)

    return comparison, best_name, best_model


def export_results(target_name: str, comparison: pd.DataFrame, best_name: str, best_model):
    out = safe_name(target_name)
    comparison.to_csv(OUTPUT_DIR / f"{out}_macro_comparison.csv", index=False)

    if best_model is not None:
        coef_table = pd.DataFrame({
            "variable": best_model.params.index,
            "coef": best_model.params.values,
            "std_err": best_model.bse.values,
            "t_stat": best_model.tvalues.values,
            "p_value": best_model.pvalues.values,
        })
        coef_table.to_csv(OUTPUT_DIR / f"{out}_macro_best_coefficients.csv", index=False)

        with open(OUTPUT_DIR / f"{out}_macro_best_summary.txt", "w") as f:
            f.write(best_model.summary().as_text())

        pd.DataFrame([{
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_r_squared": best_model.rsquared,
            "best_adj_r_squared": best_model.rsquared_adj,
            "aic": best_model.aic,
            "bic": best_model.bic,
            "n_obs": int(best_model.nobs),
        }]).to_csv(OUTPUT_DIR / f"{out}_macro_best_metrics.csv", index=False)


# =========================================================
# Feature engineering
# =========================================================
monthly_reg = build_positive_oil_features(monthly_reg, "Brent_monthly_ret", "brent_monthly")
quarterly_reg = build_positive_oil_features(quarterly_reg, "Brent_quarterly_ret", "brent_quarterly")

monthly_targets = [
    "Industrial production_growth",
    "US Retail Sales_growth",
    "Manufacturing ISM_chg",
    "Manufacturing ISM - Price Paid_chg",
]

quarterly_targets = [
    "US GDP_growth",
    "GDP - Personal Consumption_growth",
    "GDP - Investment_growth",
]

monthly_reg = create_t1_targets(monthly_reg, monthly_targets)
quarterly_reg = create_t1_targets(quarterly_reg, quarterly_targets)

# add lags for macro variables
monthly_reg = add_lags(
    monthly_reg,
    [
        "Industrial production_growth",
        "US Retail Sales_growth",
        "Manufacturing ISM_chg",
        "Manufacturing ISM - Price Paid_chg",
        "CFNAI Index_chg",
    ],
    lags=(1,)
)

quarterly_reg = add_lags(
    quarterly_reg,
    [
        "US GDP_growth",
        "GDP - Personal Consumption_growth",
        "GDP - Investment_growth",
    ],
    lags=(1,)
)


# =========================================================
# Monthly model specs
# =========================================================
monthly_specs_map = {
    "Industrial production_growth_t1": {
        "m1_baseline": [
            "brent_monthly_ret_pos",
            "Industrial production_growth_lag1",
        ],
        "m2_large_shock": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Industrial production_growth_lag1",
        ],
        "m3_oil_lags": [
            "brent_monthly_ret_pos",
            "brent_monthly_ret_pos_lag1",
            "brent_monthly_ret_pos_lag2",
            "Industrial production_growth_lag1",
        ],
        "m4_with_ism": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Industrial production_growth_lag1",
            "Manufacturing ISM_chg_lag1",
        ],
        "m5_full_macro": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "brent_monthly_ret_pos_lag1",
            "Industrial production_growth_lag1",
            "Manufacturing ISM_chg_lag1",
            "US Retail Sales_growth_lag1",
            "CFNAI Index_chg_lag1",
        ],
    },

    "US Retail Sales_growth_t1": {
        "m1_baseline": [
            "brent_monthly_ret_pos",
            "US Retail Sales_growth_lag1",
        ],
        "m2_large_shock": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "US Retail Sales_growth_lag1",
        ],
        "m3_oil_lags": [
            "brent_monthly_ret_pos",
            "brent_monthly_ret_pos_lag1",
            "brent_monthly_ret_pos_lag2",
            "US Retail Sales_growth_lag1",
        ],
        "m4_with_cfna": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "US Retail Sales_growth_lag1",
            "CFNAI Index_chg_lag1",
        ],
        "m5_full_macro": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "brent_monthly_ret_pos_lag1",
            "US Retail Sales_growth_lag1",
            "Manufacturing ISM_chg_lag1",
            "CFNAI Index_chg_lag1",
        ],
    },

    "Manufacturing ISM_chg_t1": {
        "m1_baseline": [
            "brent_monthly_ret_pos",
            "Manufacturing ISM_chg_lag1",
        ],
        "m2_large_shock": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Manufacturing ISM_chg_lag1",
        ],
        "m3_oil_lags": [
            "brent_monthly_ret_pos",
            "brent_monthly_ret_pos_lag1",
            "Manufacturing ISM_chg_lag1",
        ],
        "m4_price_paid": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Manufacturing ISM_chg_lag1",
            "Manufacturing ISM - Price Paid_chg_lag1",
        ],
        "m5_full_macro": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "brent_monthly_ret_pos_lag1",
            "Manufacturing ISM_chg_lag1",
            "Manufacturing ISM - Price Paid_chg_lag1",
            "CFNAI Index_chg_lag1",
            "US Retail Sales_growth_lag1",
        ],
    },

    "Manufacturing ISM - Price Paid_chg_t1": {
        "m1_baseline": [
            "brent_monthly_ret_pos",
            "Manufacturing ISM - Price Paid_chg_lag1",
        ],
        "m2_large_shock": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Manufacturing ISM - Price Paid_chg_lag1",
        ],
        "m3_oil_lags": [
            "brent_monthly_ret_pos",
            "brent_monthly_ret_pos_lag1",
            "Manufacturing ISM - Price Paid_chg_lag1",
        ],
        "m4_with_ism": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "Manufacturing ISM - Price Paid_chg_lag1",
            "Manufacturing ISM_chg_lag1",
        ],
        "m5_full_macro": [
            "brent_monthly_ret_pos",
            "brent_monthly_large_pos_shock",
            "brent_monthly_ret_pos_lag1",
            "Manufacturing ISM - Price Paid_chg_lag1",
            "Manufacturing ISM_chg_lag1",
            "CFNAI Index_chg_lag1",
        ],
    },
}


# =========================================================
# Quarterly model specs
# =========================================================
quarterly_specs_map = {
    "US GDP_growth_t1": {
        "q1_baseline": [
            "brent_quarterly_ret_pos",
            "US GDP_growth_lag1",
        ],
        "q2_large_shock": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "US GDP_growth_lag1",
        ],
        "q3_oil_lag": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_ret_pos_lag1",
            "US GDP_growth_lag1",
        ],
        "q4_full": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "brent_quarterly_ret_pos_lag1",
            "US GDP_growth_lag1",
        ],
    },

    "GDP - Personal Consumption_growth_t1": {
        "q1_baseline": [
            "brent_quarterly_ret_pos",
            "GDP - Personal Consumption_growth_lag1",
        ],
        "q2_large_shock": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "GDP - Personal Consumption_growth_lag1",
        ],
        "q3_oil_lag": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_ret_pos_lag1",
            "GDP - Personal Consumption_growth_lag1",
        ],
        "q4_with_gdp": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "GDP - Personal Consumption_growth_lag1",
            "US GDP_growth_lag1",
        ],
    },

    "GDP - Investment_growth_t1": {
        "q1_baseline": [
            "brent_quarterly_ret_pos",
            "GDP - Investment_growth_lag1",
        ],
        "q2_large_shock": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "GDP - Investment_growth_lag1",
        ],
        "q3_oil_lag": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_ret_pos_lag1",
            "GDP - Investment_growth_lag1",
        ],
        "q4_with_gdp": [
            "brent_quarterly_ret_pos",
            "brent_quarterly_large_pos_shock",
            "GDP - Investment_growth_lag1",
            "US GDP_growth_lag1",
        ],
    },
}


# =========================================================
# Run model selection
# =========================================================
summary_rows = []

# Monthly
for target_name, specs in monthly_specs_map.items():
    print("\n" + "=" * 100)
    print(f"MONTHLY TARGET: {target_name}")
    print("=" * 100)

    comparison, best_name, best_model = compare_models(monthly_reg, target_name, specs, hac_lags=3)
    export_results(target_name, comparison, best_name, best_model)

    print(comparison[["model_name", "n_obs", "r_squared", "adj_r_squared", "aic", "bic"]])

    if best_model is not None:
        print(f"\nBest monthly model for {target_name}: {best_name}")
        print(best_model.summary())

        summary_rows.append({
            "frequency": "monthly",
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_r_squared": best_model.rsquared,
            "best_adj_r_squared": best_model.rsquared_adj,
            "aic": best_model.aic,
            "bic": best_model.bic,
            "n_obs": int(best_model.nobs),
        })

# Quarterly
for target_name, specs in quarterly_specs_map.items():
    print("\n" + "=" * 100)
    print(f"QUARTERLY TARGET: {target_name}")
    print("=" * 100)

    comparison, best_name, best_model = compare_models(quarterly_reg, target_name, specs, hac_lags=1)
    export_results(target_name, comparison, best_name, best_model)

    print(comparison[["model_name", "n_obs", "r_squared", "adj_r_squared", "aic", "bic"]])

    if best_model is not None:
        print(f"\nBest quarterly model for {target_name}: {best_name}")
        print(best_model.summary())

        summary_rows.append({
            "frequency": "quarterly",
            "dependent_variable": target_name,
            "best_model_name": best_name,
            "best_r_squared": best_model.rsquared,
            "best_adj_r_squared": best_model.rsquared_adj,
            "aic": best_model.aic,
            "bic": best_model.bic,
            "n_obs": int(best_model.nobs),
        })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["frequency", "best_r_squared"],
    ascending=[True, False]
).reset_index(drop=True)

summary_df.to_csv(OUTPUT_DIR / "best_macro_models_summary.csv", index=False)

print("\n" + "=" * 100)
print("BEST MACRO MODELS SUMMARY")
print("=" * 100)
print(summary_df)