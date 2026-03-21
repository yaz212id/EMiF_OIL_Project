import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = BASE_DIR / "Data" / "interim"
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load clean datasets
# =========================================================
daily = pd.read_csv(INTERIM_DIR / "daily_clean.csv", parse_dates=["Date"])
monthly = pd.read_csv(INTERIM_DIR / "monthly_clean.csv", parse_dates=["Date"])
quarterly = pd.read_csv(INTERIM_DIR / "quarterly_clean.csv", parse_dates=["Date"])

# Extra safety
daily = daily.sort_values("Date").reset_index(drop=True)
monthly = monthly.sort_values("Date").reset_index(drop=True)
quarterly = quarterly.sort_values("Date").reset_index(drop=True)


# =========================================================
# DAILY TRANSFORMATIONS
# =========================================================
# Price-like series -> log returns (%)
daily_price_cols = [
    "Brent futures",
    "S&P500",
    "MSCI World",
    "Gold",
]

# Yield / rate series -> first differences
daily_rate_cols = [
    "US 10-year Rate",
    "US 2-year Rate",
    "High yield index",
]

for col in daily_price_cols:
    if col in daily.columns:
        daily[f"{col}_ret"] = 100 * np.log(daily[col] / daily[col].shift(1))

for col in daily_rate_cols:
    if col in daily.columns:
        daily[f"{col}_chg"] = daily[col] - daily[col].shift(1)

# Oil shock variables based on Brent
if "Brent futures_ret" in daily.columns:
    daily["oil_pos_shock"] = (daily["Brent futures_ret"] > 0).astype(int)

    pos_ret = daily.loc[daily["Brent futures_ret"] > 0, "Brent futures_ret"].dropna()
    if len(pos_ret) > 0:
        threshold = pos_ret.quantile(0.90)
        daily["oil_large_shock"] = (daily["Brent futures_ret"] >= threshold).astype(int)
    else:
        daily["oil_large_shock"] = 0

# One-step-ahead targets for predictive regressions
if "S&P500_ret" in daily.columns:
    daily["S&P500_ret_t1"] = daily["S&P500_ret"].shift(-1)

if "US 10-year Rate_chg" in daily.columns:
    daily["US 10-year Rate_chg_t1"] = daily["US 10-year Rate_chg"].shift(-1)

if "High yield index_chg" in daily.columns:
    daily["High yield index_chg_t1"] = daily["High yield index_chg"].shift(-1)

# Keep model-ready rows
daily_model = daily.dropna().reset_index(drop=True)


# =========================================================
# MONTHLY TRANSFORMATIONS
# =========================================================
# Level-like macro variables -> log growth (%)
monthly_growth_cols = [
    "Industrial production",
    "US Retail Sales",
]

# Survey / diffusion variables -> simple changes
monthly_change_cols = [
    "CFNAI Index",
    "Manufacturing ISM",
    "Manufacturing ISM - Price Paid",
    "Service ISM",
    "Service ISM - Price Paid",
]

for col in monthly_growth_cols:
    if col in monthly.columns:
        monthly[f"{col}_growth"] = 100 * np.log(monthly[col] / monthly[col].shift(1))

for col in monthly_change_cols:
    if col in monthly.columns:
        monthly[f"{col}_chg"] = monthly[col] - monthly[col].shift(1)

monthly_model = monthly.dropna().reset_index(drop=True)


# =========================================================
# QUARTERLY TRANSFORMATIONS
# =========================================================
quarterly_growth_cols = [
    "US GDP",
    "GDP - Personal Consumption",
    "GDP - Goods Consumption",
    "GDP - Durable Goods Consumption",
    "GDP - Non durable Goods",
    "GDP - Service",
    "GDP - Investment",
]

for col in quarterly_growth_cols:
    if col in quarterly.columns:
        quarterly[f"{col}_growth"] = 100 * np.log(quarterly[col] / quarterly[col].shift(1))

quarterly_model = quarterly.dropna().reset_index(drop=True)


# =========================================================
# Save processed datasets
# =========================================================
daily_out = PROCESSED_DIR / "daily_model.csv"
monthly_out = PROCESSED_DIR / "monthly_model.csv"
quarterly_out = PROCESSED_DIR / "quarterly_model.csv"

daily_model.to_csv(daily_out, index=False)
monthly_model.to_csv(monthly_out, index=False)
quarterly_model.to_csv(quarterly_out, index=False)

print("Processed files created successfully.\n")
print(f"Saved: {daily_out}")
print(f"Saved: {monthly_out}")
print(f"Saved: {quarterly_out}")

print("\nShapes:")
print("daily_model    :", daily_model.shape)
print("monthly_model  :", monthly_model.shape)
print("quarterly_model:", quarterly_model.shape)

print("\nDaily model preview:")
print(daily_model.head())

print("\nMonthly model preview:")
print(monthly_model.head())

print("\nQuarterly model preview:")
print(quarterly_model.head())