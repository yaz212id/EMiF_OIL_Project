from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# =========================================================
# 0) CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

START_DATE = "2010-01-01"
END_DATE = "2019-12-31"

MONTHLY_TARGETS = {
    "ip": "Industrial production_growth",
    "retail": "US Retail Sales_growth",
    "ism": "Manufacturing ISM_chg",
    "ism_price": "Manufacturing ISM - Price Paid_chg",
    "cfnai": "CFNAI Index_chg",
}

# cleaner, target-specific cross controls
MONTHLY_CROSS_CONTROLS = {
    "ip": ["ism_vol_abs", "ism_price_vol_abs", "retail_vol_abs"],
    "retail": ["ip_vol_abs", "ism_vol_abs", "cfnai_vol_abs"],
    "ism": ["ip_vol_abs", "ism_price_vol_abs", "retail_vol_abs"],
    "ism_price": ["ism_vol_abs", "ip_vol_abs", "retail_vol_abs"],
    "cfnai": ["ip_vol_abs", "retail_vol_abs", "ism_vol_abs"],
}

SOFT_SHOCK_QUANTILE = 0.80
HAC_LAGS = 2


# =========================================================
# 1) FILE LOADING
# =========================================================
def find_file(filename: str) -> Path:
    candidates = [
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT / "data" / "processed" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "data" / "processed" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} not found in project root or data/processed.")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.loc[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()
    return df


# =========================================================
# 2) FEATURE ENGINEERING
# =========================================================
def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def build_monthly_oil_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if "Brent futures_ret" not in daily.columns:
        raise KeyError("Column 'Brent futures_ret' missing from daily_model.csv")

    oil = daily[["Date", "Brent futures_ret"]].copy()
    oil["YearMonth"] = oil["Date"].dt.to_period("M")

    # monthly oil return = sum of daily returns
    monthly_oil = oil.groupby("YearMonth", as_index=False)["Brent futures_ret"].sum()
    monthly_oil["Date"] = monthly_oil["YearMonth"].dt.to_timestamp("M")
    monthly_oil = monthly_oil.rename(columns={"Brent futures_ret": "Brent_monthly_ret"})
    monthly_oil = monthly_oil[["Date", "Brent_monthly_ret"]]

    return monthly_oil


def prepare_monthly_features(monthly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()

    # merge monthly oil return
    monthly_oil = build_monthly_oil_from_daily(daily)
    df = df.merge(monthly_oil, on="Date", how="left")

    # -------------------------
    # Oil positive-return features
    # -------------------------
    df["oil_ret"] = df["Brent_monthly_ret"]
    df["oil_pos"] = df["oil_ret"].clip(lower=0)

    pos_sample = df.loc[df["oil_ret"] > 0, "oil_ret"].dropna()
    if len(pos_sample) == 0:
        raise ValueError("No positive monthly oil returns found in sample.")

    # softer threshold: more useful in calmer regime
    soft_threshold = pos_sample.quantile(SOFT_SHOCK_QUANTILE)
    df["oil_large_pos"] = (df["oil_ret"] >= soft_threshold).astype(int)

    df["oil_pos_lag1"] = df["oil_pos"].shift(1)
    df["oil_pos_lag2"] = df["oil_pos"].shift(2)

    df["oil_large_pos_lag1"] = df["oil_large_pos"].shift(1)

    df["oil_pos_x_large"] = df["oil_pos"] * df["oil_large_pos"]
    df["oil_pos_x_large_lag1"] = df["oil_pos_x_large"].shift(1)

    # smoothed positive oil pressure
    df["oil_pos_ma3"] = df["oil_pos"].rolling(3).mean()
    df["oil_pos_ma5"] = df["oil_pos"].rolling(5).mean()
    df["oil_pos_ma3_lag1"] = df["oil_pos_ma3"].shift(1)
    df["oil_pos_ma5_lag1"] = df["oil_pos_ma5"].shift(1)

    # optional oil-volatility robustness
    df["oil_vol_abs"] = df["oil_ret"].abs()
    df["oil_vol_abs_lag1"] = df["oil_vol_abs"].shift(1)

    # for figure only
    df["oil_vol_z"] = zscore(df["oil_vol_abs"])

    # -------------------------
    # Macro volatility proxies
    # -------------------------
    for short_name, col in MONTHLY_TARGETS.items():
        if col not in df.columns:
            raise KeyError(f"Missing monthly column: {col}")

        # main proxy
        df[f"{short_name}_vol_abs"] = df[col].abs()

        # robustness proxy
        df[f"{short_name}_vol_sq"] = df[col] ** 2

        # lagged main proxy
        df[f"{short_name}_vol_abs_lag1"] = df[f"{short_name}_vol_abs"].shift(1)
        df[f"{short_name}_vol_sq_lag1"] = df[f"{short_name}_vol_sq"].shift(1)

        # forecasting targets
        df[f"{short_name}_vol_abs_t1"] = df[f"{short_name}_vol_abs"].shift(-1)
        df[f"{short_name}_vol_sq_t1"] = df[f"{short_name}_vol_sq"].shift(-1)

        # monthly HAR-style persistence
        df[f"{short_name}_vol_abs_3m"] = df[f"{short_name}_vol_abs"].rolling(3).mean().shift(1)
        df[f"{short_name}_vol_abs_6m"] = df[f"{short_name}_vol_abs"].rolling(6).mean().shift(1)

        # standardized for figure
        df[f"{short_name}_vol_z"] = zscore(df[f"{short_name}_vol_abs"])

    return df


# =========================================================
# 3) HELPERS
# =========================================================
def fit_ols_hac(data: pd.DataFrame, y_col: str, x_cols: list[str], hac_lags: int = HAC_LAGS):
    used_x = [x for x in x_cols if x in data.columns]
    tmp = data[[y_col] + used_x].dropna().copy()
    if tmp.empty:
        return None, None, None

    X = sm.add_constant(tmp[used_x])
    y = tmp[y_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model, tmp, used_x


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_common_sample_models(df: pd.DataFrame, y_col: str, specs: dict[str, list[str]], hac_lags: int = HAC_LAGS):
    """
    One common sample per target across all candidate models.
    This makes adj R² / RMSE / MAE / BIC much more comparable.
    """
    union_x = sorted(set(x for cols in specs.values() for x in cols if x in df.columns))
    common = df[[y_col] + union_x].dropna().copy()

    results = []
    if common.empty:
        return results

    for model_name, x_cols in specs.items():
        used_x = [x for x in x_cols if x in common.columns]
        tmp = common[[y_col] + used_x].dropna().copy()
        if tmp.empty:
            continue

        X = sm.add_constant(tmp[used_x])
        y = tmp[y_col]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        y_hat = model.predict(X)

        results.append({
            "model_name": model_name,
            "model": model,
            "tmp": tmp,
            "used_x": used_x,
            "y_hat": y_hat,
        })

    return results


# =========================================================
# 4) MONTHLY VOLATILITY MODEL COMPARISON
# =========================================================
def build_monthly_abs_specs(short_name: str) -> dict[str, list[str]]:
    own_lag = f"{short_name}_vol_abs_lag1"
    own_3m = f"{short_name}_vol_abs_3m"
    own_6m = f"{short_name}_vol_abs_6m"
    cross = MONTHLY_CROSS_CONTROLS.get(short_name, [])

    return {
        # 1) simple persistence benchmark
        "m_abs_ar1": [own_lag],

        # 2) monthly HAR benchmark
        "m_abs_har": [own_lag, own_3m, own_6m],

        # 3) HAR + positive oil pressure
        "m_abs_har_oil_pos": [
            own_lag, own_3m, own_6m,
            "oil_pos_lag1",
            "oil_pos_ma3_lag1",
            "oil_pos_ma5_lag1",
        ],

        # 4) HAR + oil pressure + cross-macro volatility
        "m_abs_har_oil_pos_cross": [
            own_lag, own_3m, own_6m,
            "oil_pos_lag1",
            "oil_pos_ma3_lag1",
            "oil_pos_ma5_lag1",
        ] + cross,

        # 5) robustness with soft big shock
        "m_abs_har_soft_bigshock": [
            own_lag, own_3m, own_6m,
            "oil_pos_lag1",
            "oil_large_pos_lag1",
            "oil_pos_x_large_lag1",
        ],

        # 6) robustness with oil volatility
        "m_abs_har_oil_vol": [
            own_lag, own_3m, own_6m,
            "oil_vol_abs_lag1",
            "oil_pos_ma3_lag1",
        ],
    }


def run_monthly_volatility_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows = []
    coef_rows = []

    for short_name, target_col in MONTHLY_TARGETS.items():
        abs_target = f"{short_name}_vol_abs_t1"
        abs_specs = build_monthly_abs_specs(short_name)

        results = fit_common_sample_models(df, abs_target, abs_specs, hac_lags=HAC_LAGS)

        for res in results:
            model_name = res["model_name"]
            model = res["model"]
            tmp = res["tmp"]
            used_x = res["used_x"]
            y_true = tmp[abs_target]
            y_hat = res["y_hat"]

            perf_rows.append({
                "target_macro": target_col,
                "target_volatility": abs_target,
                "model": model_name,
                "n_obs": int(model.nobs),
                "n_regressors": len(used_x),
                "adj_r_squared": model.rsquared_adj,
                "r_squared": model.rsquared,
                "aic": model.aic,
                "bic": model.bic,
                "rmse": rmse(y_true, y_hat),
                "mae": mae(y_true, y_hat),
                "oil_pos_coef": model.params.get("oil_pos_lag1", np.nan),
                "oil_pos_pvalue": model.pvalues.get("oil_pos_lag1", np.nan),
                "oil_ma3_coef": model.params.get("oil_pos_ma3_lag1", np.nan),
                "oil_ma3_pvalue": model.pvalues.get("oil_pos_ma3_lag1", np.nan),
                "oil_ma5_coef": model.params.get("oil_pos_ma5_lag1", np.nan),
                "oil_ma5_pvalue": model.pvalues.get("oil_pos_ma5_lag1", np.nan),
                "oil_bigshock_coef": model.params.get("oil_large_pos_lag1", np.nan),
                "oil_bigshock_pvalue": model.pvalues.get("oil_large_pos_lag1", np.nan),
                "regressors": " | ".join(used_x),
            })

            for var in model.params.index:
                coef_rows.append({
                    "target_macro": target_col,
                    "target_volatility": abs_target,
                    "model": model_name,
                    "variable": var,
                    "coef": model.params[var],
                    "std_err": model.bse[var],
                    "t_stat": model.tvalues[var],
                    "p_value": model.pvalues[var],
                })

    perf = pd.DataFrame(perf_rows)
    coef = pd.DataFrame(coef_rows)

    # easier ranking for notebook use
    perf = perf.sort_values(
        ["target_macro", "adj_r_squared", "rmse", "mae", "bic"],
        ascending=[True, False, True, True, True]
    ).reset_index(drop=True)

    best = perf.groupby("target_macro", as_index=False).first()

    # compact summary table for notebook
    compact = best[[
        "target_macro",
        "model",
        "adj_r_squared",
        "rmse",
        "mae",
        "oil_pos_pvalue",
        "oil_ma3_pvalue",
        "oil_ma5_pvalue",
        "oil_bigshock_pvalue",
    ]].copy()

    perf.to_csv(OUTPUT_TABLES / "monthly_volatility_model_comparison.csv", index=False)
    best.to_csv(OUTPUT_TABLES / "monthly_volatility_best_models.csv", index=False)
    compact.to_csv(OUTPUT_TABLES / "monthly_volatility_compact_summary.csv", index=False)
    coef.to_csv(OUTPUT_TABLES / "monthly_volatility_models_coefficients.csv", index=False)

    return perf, best, compact


# =========================================================
# 5) FIGURES
# =========================================================
def save_monthly_figures(df: pd.DataFrame) -> None:
    start_label = df["Date"].min().date()
    end_label = df["Date"].max().date()

    # Positive monthly oil returns
    plt.figure(figsize=(11, 4))
    plt.plot(df["Date"], df["oil_pos"])
    plt.title(f"Positive monthly Brent returns ({start_label} to {end_label})")
    plt.xlabel("Date")
    plt.ylabel("Positive monthly Brent return")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "monthly_oil_positive_returns.png", dpi=160)
    plt.close()

    # Example HAR components on industrial production volatility
    if {"ip_vol_abs", "ip_vol_abs_3m", "ip_vol_abs_6m"}.issubset(df.columns):
        plt.figure(figsize=(11, 4))
        plt.plot(df["Date"], df["ip_vol_abs"], label="IP abs volatility", linewidth=1.5)
        plt.plot(df["Date"], df["ip_vol_abs_3m"], label="3-month avg lagged", alpha=0.85)
        plt.plot(df["Date"], df["ip_vol_abs_6m"], label="6-month avg lagged", alpha=0.85)
        plt.title("Industrial production volatility proxy and monthly HAR-style components")
        plt.xlabel("Date")
        plt.ylabel("Absolute macro change proxy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES / "monthly_ip_vol_har_components.png", dpi=160)
        plt.close()

    # Oil volatility vs monthly macro volatilities
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["oil_vol_z"], label="Oil volatility (z-score)", linewidth=2)
    for short_name in MONTHLY_TARGETS.keys():
        plt.plot(df["Date"], df[f"{short_name}_vol_z"], label=f"{short_name.upper()} volatility (z-score)", alpha=0.85)

    plt.title("Standardized monthly volatility comparison: Oil vs macro variables")
    plt.xlabel("Date")
    plt.ylabel("Standardized absolute change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "monthly_oil_vs_macro_volatility_curve.png", dpi=160)
    plt.close()


# =========================================================
# 6) MAIN
# =========================================================
def main():
    daily_path = find_file("daily_model.csv")
    monthly_path = find_file("monthly_model.csv")

    daily = load_csv(daily_path)
    monthly = load_csv(monthly_path)

    df = prepare_monthly_features(monthly, daily)

    # Step 1: estimate monthly volatility models
    vol_perf, vol_best, vol_compact = run_monthly_volatility_model_comparison(df)

    # Step 2: save figures
    save_monthly_figures(df)

    print("\nFiles used:")
    print("-", daily_path)
    print("-", monthly_path)

    print("\nSaved tables:")
    print("-", OUTPUT_TABLES / "monthly_volatility_model_comparison.csv")
    print("-", OUTPUT_TABLES / "monthly_volatility_best_models.csv")
    print("-", OUTPUT_TABLES / "monthly_volatility_compact_summary.csv")
    print("-", OUTPUT_TABLES / "monthly_volatility_models_coefficients.csv")

    print("\nSaved figures:")
    print("-", OUTPUT_FIGURES / "monthly_oil_positive_returns.png")
    print("-", OUTPUT_FIGURES / "monthly_ip_vol_har_components.png")
    print("-", OUTPUT_FIGURES / "monthly_oil_vs_macro_volatility_curve.png")

    print("\nCompact monthly best-model summary:")
    print(vol_compact.round(4))


if __name__ == "__main__":
    main()