from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.arima.model import ARIMA


# =========================================================
# 0) CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# Focus on calmer oil regime
START_DATE = "2010-01-01"
END_DATE = "2019-12-31"

OIL_COL = "Brent futures_ret"

TARGETS = {
    "spx": "S&P500_ret",
    "us10y": "US 10-year Rate_chg",
    "hy": "High yield index_chg",
    "gold": "Gold_ret",
}

VOL_CROSS_CONTROLS = {
    "spx": ["us10y_vol_abs", "hy_vol_abs", "gold_vol_abs"],
    "us10y": ["spx_vol_abs", "hy_vol_abs", "gold_vol_abs"],
    "hy": ["spx_vol_abs", "us10y_vol_abs", "gold_vol_abs"],
    "gold": ["spx_vol_abs", "us10y_vol_abs", "hy_vol_abs"],
}

HAC_LAGS = 5
SOFT_SHOCK_QUANTILE = 0.85


# =========================================================
# 1) FILE LOADING
# =========================================================
def find_daily_file() -> Path:
    candidates = [
        PROJECT_ROOT / "daily_model.csv",
        PROJECT_ROOT / "data" / "daily_model.csv",
        PROJECT_ROOT / "data" / "processed" / "daily_model.csv",
        Path.cwd() / "daily_model.csv",
        Path.cwd() / "data" / "daily_model.csv",
        Path.cwd() / "data" / "processed" / "daily_model.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "daily_model.csv not found. Put it in the project root or in data/processed."
    )


def load_data(path: Path) -> pd.DataFrame:
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if OIL_COL not in df.columns:
        raise KeyError(f"Missing oil column: {OIL_COL}")

    # -------------------------
    # Oil return features
    # -------------------------
    df["oil_ret"] = df[OIL_COL]

    # only positive oil returns
    df["oil_pos"] = df["oil_ret"].clip(lower=0)

    # softer large-shock dummy for 2010–2019
    pos_sample = df.loc[df["oil_ret"] > 0, "oil_ret"].dropna()
    if len(pos_sample) == 0:
        raise ValueError("No positive oil returns found in the sample.")

    soft_threshold = pos_sample.quantile(SOFT_SHOCK_QUANTILE)
    df["oil_large_pos"] = (df["oil_ret"] >= soft_threshold).astype(int)

    # lags
    df["oil_pos_lag1"] = df["oil_pos"].shift(1)
    df["oil_pos_lag2"] = df["oil_pos"].shift(2)

    df["oil_large_pos_lag1"] = df["oil_large_pos"].shift(1)

    # nonlinear interaction
    df["oil_pos_x_large"] = df["oil_pos"] * df["oil_large_pos"]
    df["oil_pos_x_large_lag1"] = df["oil_pos_x_large"].shift(1)

    # smoothed oil pressure
    df["oil_pos_ma3"] = df["oil_pos"].rolling(3).mean()
    df["oil_pos_ma5"] = df["oil_pos"].rolling(5).mean()
    df["oil_pos_ma3_lag1"] = df["oil_pos_ma3"].shift(1)
    df["oil_pos_ma5_lag1"] = df["oil_pos_ma5"].shift(1)

    # oil volatility proxy for figures
    df["oil_vol_abs"] = df["oil_ret"].abs()
    df["oil_vol_z"] = zscore(df["oil_vol_abs"])

    # -------------------------
    # Financial asset features
    # -------------------------
    for short_name, col in TARGETS.items():
        if col not in df.columns:
            raise KeyError(f"Missing target column: {col}")

        # returns
        df[f"{short_name}_ret_lag1"] = df[col].shift(1)

        # volatility proxies
        df[f"{short_name}_vol_abs"] = df[col].abs()
        df[f"{short_name}_vol_sq"] = df[col] ** 2

        # lagged volatility
        df[f"{short_name}_vol_abs_lag1"] = df[f"{short_name}_vol_abs"].shift(1)
        df[f"{short_name}_vol_sq_lag1"] = df[f"{short_name}_vol_sq"].shift(1)

        # target at t+1
        df[f"{short_name}_vol_abs_t1"] = df[f"{short_name}_vol_abs"].shift(-1)
        df[f"{short_name}_vol_sq_t1"] = df[f"{short_name}_vol_sq"].shift(-1)

        # HAR-style persistence terms
        df[f"{short_name}_vol_abs_week"] = df[f"{short_name}_vol_abs"].rolling(5).mean().shift(1)
        df[f"{short_name}_vol_abs_month"] = df[f"{short_name}_vol_abs"].rolling(22).mean().shift(1)

        # standardized vol for figure
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
    Estimate all candidate models on the same common sample for one target.
    That makes R² / RMSE / MAE / BIC much more comparable.
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
# 4) BASELINE RETURN REGRESSIONS
# =========================================================
def run_baseline_return_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coef_rows = []

    for short_name, y_col in TARGETS.items():
        specs = {
            "baseline_oil_lags": ["oil_pos_lag1", "oil_pos_lag2"],
            "baseline_oil_smoothed": ["oil_pos_ma3_lag1", "oil_pos_ma5_lag1"],
            "baseline_oil_soft_bigshock": [
                "oil_pos_lag1",
                "oil_large_pos_lag1",
                "oil_pos_x_large_lag1",
            ],
        }

        for model_name, x_cols in specs.items():
            model, tmp, used_x = fit_ols_hac(df, y_col, x_cols, hac_lags=HAC_LAGS)
            if model is None:
                continue

            X = sm.add_constant(tmp[used_x])
            y_true = tmp[y_col]
            y_hat = model.predict(X)

            rows.append({
                "target": y_col,
                "model": model_name,
                "n_obs": int(model.nobs),
                "n_regressors": len(used_x),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "rmse": rmse(y_true, y_hat),
                "mae": mae(y_true, y_hat),
                "regressors": " | ".join(used_x),
            })

            for var in model.params.index:
                coef_rows.append({
                    "target": y_col,
                    "model": model_name,
                    "variable": var,
                    "coef": model.params[var],
                    "std_err": model.bse[var],
                    "t_stat": model.tvalues[var],
                    "p_value": model.pvalues[var],
                })

    perf = pd.DataFrame(rows)
    coef = pd.DataFrame(coef_rows)

    perf = perf.sort_values(
        ["target", "adj_r_squared", "rmse", "mae", "bic"],
        ascending=[True, False, True, True, True]
    ).reset_index(drop=True)

    best = perf.groupby("target", as_index=False).first()

    perf.to_csv(OUTPUT_TABLES / "baseline_return_models_performance.csv", index=False)
    best.to_csv(OUTPUT_TABLES / "baseline_return_models_best.csv", index=False)
    coef.to_csv(OUTPUT_TABLES / "baseline_return_models_coefficients.csv", index=False)

    return perf


# =========================================================
# 5) ARMA ON RETURNS ONLY
# =========================================================
def run_arma_return_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    arma_specs = {
        "white_noise": (0, 0),
        "ar1": (1, 0),
        "ma1": (0, 1),
        "arma11": (1, 1),
        "arma21": (2, 1),
    }

    for short_name, y_col in TARGETS.items():
        series = df[y_col].dropna().copy()
        if len(series) < 100:
            continue

        for model_name, (p, q) in arma_specs.items():
            try:
                model = ARIMA(series, order=(p, 0, q), trend="c").fit()

                fitted = model.fittedvalues
                aligned = pd.concat(
                    [series.rename("y_true"), fitted.rename("y_hat")],
                    axis=1
                ).dropna()

                y_true = aligned["y_true"]
                y_hat = aligned["y_hat"]

                rows.append({
                    "target": y_col,
                    "model": model_name,
                    "p": p,
                    "q": q,
                    "n_obs": len(series),
                    "aic": model.aic,
                    "bic": model.bic,
                    "rmse": rmse(y_true, y_hat),
                    "mae": mae(y_true, y_hat),
                    "ar1_pvalue": model.pvalues.get("ar.L1", np.nan),
                    "ma1_pvalue": model.pvalues.get("ma.L1", np.nan),
                })
            except Exception:
                rows.append({
                    "target": y_col,
                    "model": model_name,
                    "p": p,
                    "q": q,
                    "n_obs": len(series),
                    "aic": np.nan,
                    "bic": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "ar1_pvalue": np.nan,
                    "ma1_pvalue": np.nan,
                })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["target", "bic", "aic", "rmse", "mae"],
        ascending=[True, True, True, True, True]
    ).reset_index(drop=True)

    best = out.groupby("target", as_index=False).first()

    out.to_csv(OUTPUT_TABLES / "arma_return_models_performance.csv", index=False)
    best.to_csv(OUTPUT_TABLES / "arma_return_models_best.csv", index=False)

    return out


# =========================================================
# 6) VOLATILITY MODEL COMPARISON
# =========================================================
def build_vol_specs(short_name: str) -> dict[str, list[str]]:
    own_abs_lag = f"{short_name}_vol_abs_lag1"
    own_week = f"{short_name}_vol_abs_week"
    own_month = f"{short_name}_vol_abs_month"
    own_sq_lag = f"{short_name}_vol_sq_lag1"

    cross_controls = VOL_CROSS_CONTROLS[short_name]

    return {
        # 1) AR(1)-vol benchmark
        "vol_abs_ar1_t1": [own_abs_lag],

        # 2) HAR benchmark
        "vol_abs_har_t1": [own_abs_lag, own_week, own_month],

        # 3) HAR + smoothed oil
        "vol_abs_har_oil_smoothed_t1": [
            own_abs_lag, own_week, own_month,
            "oil_pos_ma3_lag1", "oil_pos_ma5_lag1",
        ],

        # 4) HAR + smoothed oil + cross-asset vols
        "vol_abs_har_oil_smoothed_cross_t1": [
            own_abs_lag, own_week, own_month,
            "oil_pos_ma3_lag1", "oil_pos_ma5_lag1",
        ] + cross_controls,

        # 5) robustness with soft big shock
        "vol_abs_har_soft_bigshock_t1": [
            own_abs_lag, own_week, own_month,
            "oil_pos_lag1",
            "oil_large_pos_lag1",
            "oil_pos_x_large_lag1",
        ],

        # squared proxy robustness
        "vol_sq_ar1_oil_t1": [
            own_sq_lag,
            "oil_pos_lag1",
            "oil_pos_ma3_lag1",
        ],
    }


def run_volatility_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf_rows = []
    coef_rows = []

    for short_name, y_ret in TARGETS.items():
        abs_target = f"{short_name}_vol_abs_t1"
        abs_specs = build_vol_specs(short_name)

        abs_results = fit_common_sample_models(df, abs_target, abs_specs, hac_lags=HAC_LAGS)

        for res in abs_results:
            model_name = res["model_name"]
            model = res["model"]
            tmp = res["tmp"]
            used_x = res["used_x"]
            y_true = tmp[abs_target]
            y_hat = res["y_hat"]

            perf_rows.append({
                "target_return": y_ret,
                "target_volatility": abs_target,
                "model": model_name,
                "n_obs": int(model.nobs),
                "n_regressors": len(used_x),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "rmse": rmse(y_true, y_hat),
                "mae": mae(y_true, y_hat),
                "oil_lag1_coef": model.params.get("oil_pos_lag1", np.nan),
                "oil_lag1_pvalue": model.pvalues.get("oil_pos_lag1", np.nan),
                "oil_smoothed_coef": model.params.get("oil_pos_ma3_lag1", np.nan),
                "oil_smoothed_pvalue": model.pvalues.get("oil_pos_ma3_lag1", np.nan),
                "oil_bigshock_lag1_coef": model.params.get("oil_large_pos_lag1", np.nan),
                "oil_bigshock_lag1_pvalue": model.pvalues.get("oil_large_pos_lag1", np.nan),
                "regressors": " | ".join(used_x),
            })

            for var in model.params.index:
                coef_rows.append({
                    "target_volatility": abs_target,
                    "model": model_name,
                    "variable": var,
                    "coef": model.params[var],
                    "std_err": model.bse[var],
                    "t_stat": model.tvalues[var],
                    "p_value": model.pvalues[var],
                })

        # squared-vol robustness
        sq_target = f"{short_name}_vol_sq_t1"
        sq_specs = {
            "vol_sq_ar1_oil_t1": [
                f"{short_name}_vol_sq_lag1",
                "oil_pos_lag1",
                "oil_pos_ma3_lag1",
            ]
        }
        sq_results = fit_common_sample_models(df, sq_target, sq_specs, hac_lags=HAC_LAGS)

        for res in sq_results:
            model_name = res["model_name"]
            model = res["model"]
            tmp = res["tmp"]
            used_x = res["used_x"]
            y_true = tmp[sq_target]
            y_hat = res["y_hat"]

            perf_rows.append({
                "target_return": y_ret,
                "target_volatility": sq_target,
                "model": model_name,
                "n_obs": int(model.nobs),
                "n_regressors": len(used_x),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "rmse": rmse(y_true, y_hat),
                "mae": mae(y_true, y_hat),
                "oil_lag1_coef": model.params.get("oil_pos_lag1", np.nan),
                "oil_lag1_pvalue": model.pvalues.get("oil_pos_lag1", np.nan),
                "oil_smoothed_coef": model.params.get("oil_pos_ma3_lag1", np.nan),
                "oil_smoothed_pvalue": model.pvalues.get("oil_pos_ma3_lag1", np.nan),
                "oil_bigshock_lag1_coef": model.params.get("oil_large_pos_lag1", np.nan),
                "oil_bigshock_lag1_pvalue": model.pvalues.get("oil_large_pos_lag1", np.nan),
                "regressors": " | ".join(used_x),
            })

            for var in model.params.index:
                coef_rows.append({
                    "target_volatility": sq_target,
                    "model": model_name,
                    "variable": var,
                    "coef": model.params[var],
                    "std_err": model.bse[var],
                    "t_stat": model.tvalues[var],
                    "p_value": model.pvalues[var],
                })

    perf = pd.DataFrame(perf_rows)
    coef = pd.DataFrame(coef_rows)

    perf = perf.sort_values(
        ["target_volatility", "adj_r_squared", "rmse", "mae", "bic"],
        ascending=[True, False, True, True, True]
    ).reset_index(drop=True)

    best = perf.groupby("target_volatility", as_index=False).first()

    perf.to_csv(OUTPUT_TABLES / "volatility_model_comparison.csv", index=False)
    best.to_csv(OUTPUT_TABLES / "volatility_best_models.csv", index=False)
    coef.to_csv(OUTPUT_TABLES / "volatility_models_coefficients.csv", index=False)

    return perf, best


# =========================================================
# 7) FIGURES
# =========================================================
def save_main_figures(df: pd.DataFrame) -> None:
    start_label = df["Date"].min().date()
    end_label = df["Date"].max().date()

    # Positive oil returns
    plt.figure(figsize=(11, 4))
    plt.plot(df["Date"], df["oil_pos"])
    plt.title(f"Positive Brent returns ({start_label} to {end_label})")
    plt.xlabel("Date")
    plt.ylabel("Positive Brent daily return")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "oil_positive_returns.png", dpi=160)
    plt.close()

    # S&P HAR components
    plt.figure(figsize=(11, 4))
    plt.plot(df["Date"], df["spx_vol_abs"], label="S&P500 abs return", linewidth=1.5)
    plt.plot(df["Date"], df["spx_vol_abs_week"], label="Weekly avg lagged", alpha=0.85)
    plt.plot(df["Date"], df["spx_vol_abs_month"], label="Monthly avg lagged", alpha=0.85)
    plt.title("S&P500 volatility proxy and HAR-style components")
    plt.xlabel("Date")
    plt.ylabel("Absolute return proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "spx_vol_har_components.png", dpi=160)
    plt.close()

    # Oil vs markets volatility
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["oil_vol_z"], label="Oil volatility (z-score)", linewidth=2)
    for short_name in TARGETS.keys():
        plt.plot(df["Date"], df[f"{short_name}_vol_z"], label=f"{short_name.upper()} volatility (z-score)", alpha=0.85)

    plt.title("Standardized volatility comparison: Oil vs financial markets")
    plt.xlabel("Date")
    plt.ylabel("Standardized absolute return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "oil_vs_markets_volatility_curve.png", dpi=160)
    plt.close()


# =========================================================
# 8) MAIN
# =========================================================
def main():
    data_path = find_daily_file()
    df = load_data(data_path)
    df = prepare_features(df)

    # Step 1: baseline returns
    baseline_perf = run_baseline_return_models(df)

    # Step 2: ARMA returns
    arma_perf = run_arma_return_models(df)

    # Step 3: volatility models
    vol_perf, vol_best = run_volatility_model_comparison(df)

    # Step 4: figures
    save_main_figures(df)

    print("\nData file used:")
    print(data_path)

    print("\nSaved tables:")
    print("-", OUTPUT_TABLES / "baseline_return_models_performance.csv")
    print("-", OUTPUT_TABLES / "baseline_return_models_best.csv")
    print("-", OUTPUT_TABLES / "baseline_return_models_coefficients.csv")
    print("-", OUTPUT_TABLES / "arma_return_models_performance.csv")
    print("-", OUTPUT_TABLES / "arma_return_models_best.csv")
    print("-", OUTPUT_TABLES / "volatility_model_comparison.csv")
    print("-", OUTPUT_TABLES / "volatility_best_models.csv")
    print("-", OUTPUT_TABLES / "volatility_models_coefficients.csv")

    print("\nSaved figures:")
    print("-", OUTPUT_FIGURES / "oil_positive_returns.png")
    print("-", OUTPUT_FIGURES / "spx_vol_har_components.png")
    print("-", OUTPUT_FIGURES / "oil_vs_markets_volatility_curve.png")

    print("\nBest baseline return model per index:")
    print(pd.read_csv(OUTPUT_TABLES / "baseline_return_models_best.csv").round(4))

    print("\nBest ARMA per return series:")
    print(pd.read_csv(OUTPUT_TABLES / "arma_return_models_best.csv").round(4))

    print("\nBest volatility model per index:")
    print(vol_best.round(4))


if __name__ == "__main__":
    main()