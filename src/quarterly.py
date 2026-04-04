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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
# =========================================================
# 0) CONFIG
# =========================================================
PROJECT_ROOT   = Path.cwd().parent
OUTPUT_TABLES  = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
FULL_START  = "1990-01-01"
FULL_END    = "2025-12-31"
FOCUS_START = "1995-01-01"
FOCUS_END   = "2005-12-31"
HAC_LAGS    = 1   # quarterly standard
VAR_MAXLAGS = 4   # BIC-selected up to 4
IRF_PERIODS = 8   # quarters for IRF
OOS_FRAC    = 0.2 # last 20 % held out for OOS evaluation
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
    "US GDP_growth":                          "US GDP",
    "GDP - Personal Consumption_growth":      "Personal Cons.",
    "GDP - Goods Consumption_growth":         "Goods Cons.",
    "GDP - Durable Goods Consumption_growth": "Durable Goods",
    "GDP - Non durable Goods_growth":         "Non-durable Goods",
    "GDP - Service_growth":                   "Services",
    "GDP - Investment_growth":                "Investment",
}
# =========================================================
# 1) LOAD & MERGE DATA  (unchanged)
# =========================================================
def load_data() -> pd.DataFrame:
    """Load quarterly GDP + daily Brent, aggregate oil to quarterly sums."""
    qpath = PROJECT_ROOT / "Data" / "processed" / "quarterly_model.csv"
    dpath = PROJECT_ROOT / "Data" / "processed" / "daily_model.csv"
    for p in [qpath, dpath]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    quarterly = (pd.read_csv(qpath, parse_dates=["Date"])
                   .sort_values("Date").reset_index(drop=True))
    daily     = (pd.read_csv(dpath,  parse_dates=["Date"])
                   .sort_values("Date").reset_index(drop=True))
    daily["Quarter"] = daily["Date"].dt.to_period("Q")
    q_oil = (daily.groupby("Quarter", as_index=False)["Brent futures_ret"]
                  .sum()
                  .rename(columns={"Brent futures_ret": "brent_q_ret"}))
    q_oil["Date"] = q_oil["Quarter"].dt.to_timestamp("Q")
    df = quarterly.merge(q_oil[["Date", "brent_q_ret"]], on="Date", how="left")
    return df.sort_values("Date").reset_index(drop=True)
# =========================================================
# 2) FEATURE ENGINEERING  (unchanged)
# =========================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_pos"] = df["brent_q_ret"].clip(lower=0)
    pos_vals  = df.loc[df["brent_q_ret"] > 0, "brent_q_ret"].dropna()
    threshold = pos_vals.quantile(0.90) if len(pos_vals) > 0 else 0.0
    df["oil_large_shock"]      = (df["brent_q_ret"] >= threshold).astype(int)
    df["oil_pos_x_large"]      = df["oil_pos"] * df["oil_large_shock"]
    df["oil_pos_lag1"]         = df["oil_pos"].shift(1)
    df["oil_pos_lag2"]         = df["oil_pos"].shift(2)
    df["oil_large_shock_lag1"] = df["oil_large_shock"].shift(1)
    df["oil_pos_x_large_lag1"] = df["oil_pos_x_large"].shift(1)
    for col in GDP_TARGETS:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)
            df[f"{col}_t1"]   = df[col].shift(-1)
    return df
# =========================================================
# 3) SHARED HELPERS
# =========================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def fit_ols_hac(df, y_col, x_cols):
    used = [c for c in x_cols if c in df.columns]
    tmp  = df[[y_col] + used].dropna().copy()
    if len(tmp) < len(used) + 2:
        return None, None
    X     = sm.add_constant(tmp[used])
    model = sm.OLS(tmp[y_col], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return model, tmp
def fit_common_sample(df, y_col, specs):
    """Fit all OLS specs on a common (union) sample."""
    all_x  = sorted({x for cols in specs.values() for x in cols if x in df.columns})
    common = df[[y_col] + all_x].dropna().copy()
    if common.empty:
        return []
    results = []
    for name, x_cols in specs.items():
        used = [c for c in x_cols if c in common.columns]
        tmp  = common[[y_col] + used].dropna()
        if tmp.empty:
            continue
        X     = sm.add_constant(tmp[used])
        y     = tmp[y_col]
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        results.append({"name": name, "model": model, "tmp": tmp,
                        "used_x": used, "y_hat": model.predict(X)})
    return results
def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)
# =========================================================
# 4) OLS MODEL COMPARISON  (unchanged)
# =========================================================
def build_ols_specs(target: str) -> dict:
    own_lag = f"{target}_lag1"
    return {
        "M1_baseline":    ["oil_pos", own_lag],
        "M2_large_shock": ["oil_pos", "oil_large_shock", own_lag],
        "M3_oil_lags":    ["oil_pos", "oil_pos_lag1", "oil_pos_lag2", own_lag],
        "M4_interaction": ["oil_pos", "oil_large_shock", "oil_pos_x_large", own_lag],
        "M5_full":        ["oil_pos", "oil_large_shock", "oil_pos_lag1",
                           "oil_pos_lag2", "oil_pos_x_large", own_lag],
    }
def run_ols_comparison(df: pd.DataFrame, window_label: str):
    perf_rows, coef_rows = [], []
    for target in GDP_TARGETS:
        y_col   = f"{target}_t1"
        specs   = build_ols_specs(target)
        results = fit_common_sample(df, y_col, specs)
        for r in results:
            m = r["model"]
            perf_rows.append({
                "window": window_label, "target": target, "model": r["name"],
                "n_obs": int(m.nobs), "n_regressors": len(r["used_x"]),
                "r_squared": m.rsquared, "adj_r_squared": m.rsquared_adj,
                "aic": m.aic, "bic": m.bic,
                "rmse": rmse(r["tmp"][y_col], r["y_hat"]),
                "mae":  mae(r["tmp"][y_col],  r["y_hat"]),
                "regressors": " | ".join(r["used_x"]),
            })
            for var in m.params.index:
                coef_rows.append({
                    "window": window_label, "target": target, "model": r["name"],
                    "variable": var, "coef": m.params[var],
                    "std_err": m.bse[var], "t_stat": m.tvalues[var],
                    "p_value": m.pvalues[var],
                    "sig": ("***" if m.pvalues[var] < 0.01 else
                            "**"  if m.pvalues[var] < 0.05 else
                            "*"   if m.pvalues[var] < 0.10 else ""),
                })
    perf = (pd.DataFrame(perf_rows)
              .sort_values(["target", "adj_r_squared"], ascending=[True, False])
              .reset_index(drop=True))
    coef = pd.DataFrame(coef_rows)
    best = perf.groupby("target", as_index=False).first()
    perf.to_csv(OUTPUT_TABLES / f"quarterly_ols_performance_{window_label}.csv", index=False)
    coef.to_csv(OUTPUT_TABLES / f"quarterly_ols_coefficients_{window_label}.csv", index=False)
    best.to_csv(OUTPUT_TABLES / f"quarterly_ols_best_{window_label}.csv",         index=False)
    return perf, coef, best
# =========================================================
# 5) AR(1) BENCHMARK  — univariate per GDP target
# =========================================================
def run_ar1_benchmark(df: pd.DataFrame, window_label: str):
    """
    Fit a pure AR(1): y_t = c + phi * y_{t-1} + eps_t
    using OLS with HAC standard errors.
    Also computes OOS metrics on the last OOS_FRAC of the sample
    via a rolling 1-step-ahead scheme.
    Returns a results dict keyed by target.
    """
    rows = []
    results = {}
    for target in GDP_TARGETS:
        series = df[target].dropna().reset_index(drop=True)
        n      = len(series)
        if n < 10:
            continue
        # ---- in-sample AR(1) ----
        y     = series.iloc[1:].values
        y_lag = series.iloc[:-1].values
        X_ar1 = sm.add_constant(y_lag)
        model = sm.OLS(y, X_ar1).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        y_hat = model.predict(X_ar1)
        # ---- OOS rolling 1-step-ahead ----
        n_train   = int(np.ceil(n * (1 - OOS_FRAC)))
        oos_preds = []
        oos_actuals = []
        for t in range(n_train, n - 1):
            y_tr   = series.iloc[:t].values
            y_t    = series.iloc[1:t].values
            y_t_l  = series.iloc[:t - 1].values
            X_t    = sm.add_constant(y_t_l, has_constant="add")
            m_t    = sm.OLS(y_t, X_t).fit()
            c_, p_ = m_t.params[0], m_t.params[1]
            pred   = c_ + p_ * series.iloc[t - 1]
            oos_preds.append(pred)
            oos_actuals.append(series.iloc[t])
        oos_rmse = rmse(oos_actuals, oos_preds) if oos_preds else np.nan
        oos_mae  = mae(oos_actuals,  oos_preds) if oos_preds else np.nan
        results[target] = {
            "model":       model,
            "y":           y,
            "y_hat_is":    y_hat,
            "oos_actuals": np.array(oos_actuals),
            "oos_preds":   np.array(oos_preds),
        }
        rows.append({
            "window":        window_label,
            "target":        target,
            "model_type":    "AR(1)",
            "n_obs":         len(y),
            "n_train":       n_train,
            "n_oos":         len(oos_preds),
            "phi_coef":      model.params[1],
            "phi_pvalue":    model.pvalues[1],
            "r_squared_is":  model.rsquared,
            "adj_r2_is":     model.rsquared_adj,
            "aic_is":        model.aic,
            "bic_is":        model.bic,
            "rmse_is":       rmse(y, y_hat),
            "mae_is":        mae(y,  y_hat),
            "rmse_oos":      oos_rmse,
            "mae_oos":       oos_mae,
        })
    ar1_df = pd.DataFrame(rows)
    ar1_df.to_csv(OUTPUT_TABLES / f"quarterly_ar1_{window_label}.csv", index=False)
    return ar1_df, results
# =========================================================
# 6) MULTIVARIATE VAR  — all GDP targets + oil
# =========================================================
def run_var_model(df: pd.DataFrame, window_label: str):
    """
    Fit a single VAR(p) on [brent_q_ret] + all 7 GDP growth series.
    Lag p selected by BIC (up to VAR_MAXLAGS).
    Computes in-sample fit and OOS rolling 1-step-ahead forecasts
    for each endogenous variable.
    Returns a results dict and a summary DataFrame.
    """
    # Build the multivariate matrix (drop rows with any NaN)
    var_cols = ["brent_q_ret"] + GDP_TARGETS
    var_data = df[var_cols].dropna().reset_index(drop=True)
    n_obs, k = len(var_data), len(var_cols)
    # Statsmodels requires: nobs - maxlags > k*maxlags  →  safe cap
    safe_max = max(1, min(VAR_MAXLAGS, (n_obs - k - 1) // (k + 1)))
    if n_obs < k + safe_max + 5:
        print("  [VAR] Not enough observations — skipping.")
        return pd.DataFrame(), {}, None
    # ---- Lag selection ----
    var_model = VAR(var_data)
    lag_order = var_model.select_order(maxlags=safe_max)
    p         = max(1, int(lag_order.bic))
    # ---- Full-sample fit ----
    fitted = var_model.fit(p)
    # ---- In-sample residuals / fit per variable ----
    is_rows  = []
    var_results = {"fitted": fitted, "p": p, "var_cols": var_cols,
                   "oos_preds": {}, "oos_actuals": {}}
    for col in var_cols:
        idx   = var_cols.index(col)
        y_is  = var_data[col].iloc[p:].values
        y_hat = fitted.fittedvalues[col].values
        is_rows.append({
            "variable":   col,
            "rmse_is":    rmse(y_is, y_hat),
            "mae_is":     mae(y_is,  y_hat),
        })
    # ---- OOS rolling 1-step-ahead ----
    n       = len(var_data)
    n_train = int(np.ceil(n * (1 - OOS_FRAC)))
    oos_pred_dict    = {col: [] for col in var_cols}
    oos_actual_dict  = {col: [] for col in var_cols}
    for t in range(n_train, n):
        train_chunk = var_data.iloc[:t]
        if len(train_chunk) < p + 2:
            continue
        try:
            m_t    = VAR(train_chunk).fit(p)
            lag_vals = train_chunk.values[-p:]  # shape (p, n_vars)
            fc     = m_t.forecast(lag_vals, steps=1)[0]  # 1-step forecast
            for j, col in enumerate(var_cols):
                oos_pred_dict[col].append(fc[j])
                oos_actual_dict[col].append(var_data[col].iloc[t])
        except Exception:
            pass
    # ---- Merge IS + OOS metrics ----
    rows = []
    for col in var_cols:
        preds   = np.array(oos_pred_dict[col])
        actuals = np.array(oos_actual_dict[col])
        oos_r   = rmse(actuals, preds) if len(preds) > 0 else np.nan
        oos_m   = mae(actuals,  preds) if len(preds) > 0 else np.nan
        is_entry = next((r for r in is_rows if r["variable"] == col), {})
        rows.append({
            "window":     window_label,
            "target":     col,
            "model_type": "VAR",
            "var_lags":   p,
            "n_obs":      len(var_data) - p,
            "n_train":    n_train,
            "n_oos":      len(preds),
            "rmse_is":    is_entry.get("rmse_is", np.nan),
            "mae_is":     is_entry.get("mae_is",  np.nan),
            "rmse_oos":   oos_r,
            "mae_oos":    oos_m,
        })
        var_results["oos_preds"][col]   = preds
        var_results["oos_actuals"][col] = actuals
    var_df = pd.DataFrame(rows)
    var_df.to_csv(OUTPUT_TABLES / f"quarterly_var_{window_label}.csv", index=False)
    print(f"\n  VAR lag selected by BIC: p = {p}")
    print(f"  VAR endogenous variables: {var_cols}")
    return var_df, var_results, fitted
# =========================================================
# 7) AR(1) vs VAR COMPARISON
# =========================================================
def compare_ar1_var(ar1_df: pd.DataFrame, var_df: pd.DataFrame,
                    window_label: str) -> pd.DataFrame:
    """
    Merge AR(1) and VAR metrics side-by-side for each GDP target.
    Columns: target | AR1_rmse_is | VAR_rmse_is | AR1_rmse_oos | VAR_rmse_oos
             | AR1_mae_oos | VAR_mae_oos | winner_oos
    """
    if ar1_df.empty or var_df.empty:
        return pd.DataFrame()
    # Keep only GDP targets (exclude oil from VAR metrics)
    var_gdp = var_df[var_df["target"].isin(GDP_TARGETS)].copy()
    ar1_sub = ar1_df[["target", "r_squared_is", "adj_r2_is",
                       "rmse_is", "mae_is", "rmse_oos", "mae_oos"]].copy()
    var_sub = var_gdp[["target", "rmse_is", "mae_is",
                        "rmse_oos", "mae_oos"]].copy()
    merged = ar1_sub.merge(var_sub, on="target", suffixes=("_AR1", "_VAR"))
    # Determine OOS winner per target
    def winner(row):
        if pd.isna(row["rmse_oos_AR1"]) or pd.isna(row["rmse_oos_VAR"]):
            return "N/A"
        return "AR(1)" if row["rmse_oos_AR1"] <= row["rmse_oos_VAR"] else "VAR"
    merged["winner_oos_rmse"] = merged.apply(winner, axis=1)
    merged["pct_improvement"] = (
        (merged["rmse_oos_AR1"] - merged["rmse_oos_VAR"]) /
        merged["rmse_oos_AR1"] * 100
    ).round(2)
    merged["window"] = window_label
    merged.to_csv(OUTPUT_TABLES / f"quarterly_ar1_vs_var_{window_label}.csv", index=False)
    # Pretty print
    print_section(f"AR(1) vs VAR COMPARISON — {window_label}")
    cols_show = ["target", "rmse_is_AR1", "rmse_is_VAR",
                 "rmse_oos_AR1", "rmse_oos_VAR",
                 "mae_oos_AR1",  "mae_oos_VAR",
                 "winner_oos_rmse", "pct_improvement"]
    print(merged[cols_show].round(4).to_string(index=False))
    ar1_wins = (merged["winner_oos_rmse"] == "AR(1)").sum()
    var_wins = (merged["winner_oos_rmse"] == "VAR").sum()
    print(f"\n  AR(1) wins on OOS RMSE: {ar1_wins} / {len(merged)} targets")
    print(f"  VAR   wins on OOS RMSE: {var_wins} / {len(merged)} targets")
    return merged
# =========================================================
# 8) GRANGER CAUSALITY + IRF  (unchanged logic)
# =========================================================
def run_granger_irf(df: pd.DataFrame, window_label: str):
    """Bivariate VAR per GDP target: Granger causality + IRF."""
    granger_rows, irf_data = [], {}
    for target in GDP_TARGETS:
        tmp = df[["brent_q_ret", target]].dropna().copy()
        if len(tmp) < 12:
            continue
        vm       = VAR(tmp)
        best_lag = max(1, int(vm.select_order(maxlags=VAR_MAXLAGS).bic))
        fitted   = vm.fit(best_lag)
        # Granger: oil -> GDP
        try:
            gc = grangercausalitytests(tmp[["brent_q_ret", target]].dropna(),
                                       maxlag=best_lag, verbose=False)
            p_o2g = gc[best_lag][0]["ssr_ftest"][1]
        except Exception:
            p_o2g = np.nan
        # Granger: GDP -> oil
        try:
            gc2   = grangercausalitytests(tmp[[target, "brent_q_ret"]].dropna(),
                                          maxlag=best_lag, verbose=False)
            p_g2o = gc2[best_lag][0]["ssr_ftest"][1]
        except Exception:
            p_g2o = np.nan
        # IRF
        try:
            irf      = fitted.irf(IRF_PERIODS)
            irf_vals = irf.irfs[:, 1, 0]
            irf_data[target] = {"irf": irf_vals}
        except Exception:
            irf_data[target] = {"irf": None}
        granger_rows.append({
            "window":            window_label,
            "target":            target,
            "var_lags_bic":      best_lag,
            "n_obs":             int(fitted.nobs),
            "p_oil_granger_gdp": p_o2g,
            "oil_causes_gdp":    "Yes" if (not np.isnan(p_o2g) and p_o2g < 0.10) else "No",
            "p_gdp_granger_oil": p_g2o,
            "gdp_causes_oil":    "Yes" if (not np.isnan(p_g2o) and p_g2o < 0.10) else "No",
        })
    granger_df = pd.DataFrame(granger_rows)
    granger_df.to_csv(OUTPUT_TABLES / f"quarterly_granger_{window_label}.csv", index=False)
    return granger_df, irf_data
# =========================================================
# 9) FIGURES
# =========================================================
def fig_time_series(df, window_label):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(df["Date"], df["brent_q_ret"], color="steelblue", alpha=0.7, width=60)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Brent Crude: Quarterly Log Returns (%)", fontsize=12)
    axes[0].set_ylabel("Log return (%)")
    colors = plt.cm.tab10.colors
    for i, col in enumerate(["US GDP_growth", "GDP - Personal Consumption_growth",
                              "GDP - Investment_growth", "GDP - Service_growth"]):
        if col in df.columns:
            axes[1].plot(df["Date"], col, label=TARGET_LABELS[col],
                         color=colors[i], linewidth=1.4, alpha=0.9)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("US GDP Components: Quarterly Growth (%)", fontsize=12)
    axes[1].set_ylabel("Log growth (%)"); axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"quarterly_time_series_{window_label}.png", dpi=160)
    plt.close(fig)
def fig_irf(irf_data, window_label):
    targets_plot = [t for t in GDP_TARGETS if irf_data.get(t, {}).get("irf") is not None]
    if not targets_plot:
        return
    ncols  = 2
    nrows  = int(np.ceil(len(targets_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    quarters = np.arange(IRF_PERIODS + 1)
    for idx, tgt in enumerate(targets_plot):
        row, col = divmod(idx, ncols)
        ax  = axes[row, col]
        irf = irf_data[tgt]["irf"]
        ax.plot(quarters, irf, color="steelblue", linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(quarters, 0, irf, where=(irf > 0), alpha=0.15, color="green")
        ax.fill_between(quarters, 0, irf, where=(irf < 0), alpha=0.15, color="red")
        ax.set_title(TARGET_LABELS[tgt], fontsize=10)
        ax.set_xlabel("Quarters after shock"); ax.set_ylabel("Response (%)")
        ax.set_xticks(quarters)
    for idx in range(len(targets_plot), nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)
    fig.suptitle(f"IRF: GDP Response to 1-Unit Brent Oil Shock — {window_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"quarterly_irf_{window_label}.png", dpi=160)
    plt.close(fig)
def fig_r2_bar(best_df, window_label):
    if best_df.empty:
        return
    labels  = [TARGET_LABELS.get(t, t) for t in best_df["target"]]
    r2_vals = best_df["adj_r_squared"].fillna(0).values
    colors  = ["#2196F3" if v >= 0 else "#F44336" for v in r2_vals]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, r2_vals, color=colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Best OLS Model Adj. R² by GDP Component — {window_label}", fontsize=12)
    ax.set_ylabel("Adjusted R²")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"quarterly_r2_bar_{window_label}.png", dpi=160)
    plt.close(fig)
def fig_ar1_vs_var(cmp_df: pd.DataFrame, window_label: str):
    """
    Grouped bar chart: OOS RMSE for AR(1) vs VAR, one group per GDP target.
    """
    if cmp_df.empty:
        return
    labels   = [TARGET_LABELS.get(t, t) for t in cmp_df["target"]]
    ar1_vals = cmp_df["rmse_oos_AR1"].fillna(0).values
    var_vals = cmp_df["rmse_oos_VAR"].fillna(0).values
    x    = np.arange(len(labels))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w / 2, ar1_vals, w, label="AR(1)", color="#1565C0", alpha=0.85)
    b2 = ax.bar(x + w / 2, var_vals,  w, label="VAR",   color="#E53935", alpha=0.85)
    # Annotate with values
    for bar in zip(list(b1) + list(b2)):
        for b in bar:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("OOS RMSE (1-step-ahead rolling)")
    ax.set_title(f"AR(1) vs VAR — Out-of-Sample RMSE by GDP Component — {window_label}",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"quarterly_ar1_vs_var_oos_{window_label}.png", dpi=160)
    plt.close(fig)
def fig_granger_heatmap(granger_df, window_label):
    if granger_df.empty:
        return
    labels = [TARGET_LABELS.get(t, t) for t in granger_df["target"]]
    p_vals = granger_df["p_oil_granger_gdp"].fillna(1.0).values
    fig, ax = plt.subplots(figsize=(11, 3))
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0, vmax=0.20)
    for i, (lbl, pv) in enumerate(zip(labels, p_vals)):
        ax.barh(0, 1, left=i, color=cmap(norm(pv)), edgecolor="white", height=0.6)
        sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
        ax.text(i + 0.5, 0, f"p={pv:.3f}{sig}", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.set_xlim(0, len(labels))
    ax.set_xticks([i + 0.5 for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"Granger Causality: Oil → GDP Component (p-value) — {window_label}",
                 fontsize=11)
    sm_obj = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_obj.set_array([])
    fig.colorbar(sm_obj, ax=ax, orientation="vertical", label="p-value", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / f"quarterly_granger_heatmap_{window_label}.png", dpi=160)
    plt.close(fig)
# =========================================================
# 10) PER-WINDOW RUNNER
# =========================================================
def run_window(df: pd.DataFrame, label: str, start: str, end: str) -> None:
    subset = (df.loc[(df["Date"] >= start) & (df["Date"] <= end)]
                .copy().reset_index(drop=True))
    print_section(f"WINDOW: {label}  [{start} – {end}]  n={len(subset)}")
    # --- 4. OLS comparison ---
    print_section(f"OLS MODEL COMPARISON — {label}")
    perf, coef, best_ols = run_ols_comparison(subset, label)
    print(best_ols[["target", "model", "n_obs", "adj_r_squared", "bic", "rmse"]].to_string(index=False))
    # --- 5. AR(1) benchmark ---
    print_section(f"AR(1) BENCHMARK — {label}")
    ar1_df, ar1_res = run_ar1_benchmark(subset, label)
    print(ar1_df[["target", "phi_coef", "phi_pvalue", "adj_r2_is",
                   "rmse_is", "rmse_oos", "mae_oos"]].round(4).to_string(index=False))
    # --- 6. VAR model ---
    print_section(f"VAR MODEL — {label}")
    var_df, var_res, var_fitted = run_var_model(subset, label)
    if not var_df.empty:
        gdp_only = var_df[var_df["target"].isin(GDP_TARGETS)]
        print(gdp_only[["target", "var_lags", "n_obs",
                         "rmse_is", "rmse_oos", "mae_oos"]].round(4).to_string(index=False))
    # --- 7. Comparison ---
    cmp_df = compare_ar1_var(ar1_df, var_df, label)
    # --- 8. Granger + IRF ---
    print_section(f"GRANGER CAUSALITY — {label}")
    granger_df, irf_data = run_granger_irf(subset, label)
    print(granger_df[["target", "var_lags_bic", "p_oil_granger_gdp",
                       "oil_causes_gdp"]].to_string(index=False))
    # --- 9. Figures ---
    fig_time_series(subset, label)
    fig_irf(irf_data, label)
    fig_r2_bar(best_ols, label)
    fig_granger_heatmap(granger_df, label)
    if not cmp_df.empty:
        fig_ar1_vs_var(cmp_df, label)
    print(f"\n  → Tables saved:  outputs/tables/quarterly_*_{label}.csv")
    print(f"  → Figures saved: outputs/figures/quarterly_*_{label}.png")
# =========================================================
# 11) MAIN
# =========================================================
def main() -> None:
    print_section("QUARTERLY ANALYSIS  —  EMiF OIL PROJECT")
    df = load_data()
    df = build_features(df)
    print(f"\nDataset: {df.shape[0]} rows | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"GDP targets : {len(GDP_TARGETS)}")
    print(f"OOS fraction: {int(OOS_FRAC*100)}%  (rolling 1-step-ahead)")
    run_window(df, label="FULL",  start=FULL_START,  end=FULL_END)
    run_window(df, label="FOCUS", start=FOCUS_START, end=FOCUS_END)
    # ---- Final cross-window summary ----
    print_section("CROSS-WINDOW SUMMARY")
    for lbl in ["FULL", "FOCUS"]:
        fp = OUTPUT_TABLES / f"quarterly_ar1_vs_var_{lbl}.csv"
        if fp.exists():
            c = pd.read_csv(fp)
            print(f"\nAR(1) vs VAR — {lbl}:")
            print(c[["target", "rmse_oos_AR1", "rmse_oos_VAR",
                      "winner_oos_rmse", "pct_improvement"]].round(4).to_string(index=False))
    print_section("DONE")
    print(f"\nAll outputs → {OUTPUT_TABLES}\n"
          f"             {OUTPUT_FIGURES}")
if __name__ == "__main__":
    main()