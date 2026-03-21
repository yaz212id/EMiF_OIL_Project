import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "Data" / "raw"
INTERIM_DIR = BASE_DIR / "Data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Variables to keep for the project
# =========================================================
DAILY_VARS = [
    "WTI futures",
    "Brent futures",
    "S&P500",
    "MSCI World",
    "US 10-year Rate",
    "US 2-year Rate",
    "High yield index",
    "Gold",
]

MONTHLY_VARS = [
    "Industrial production",
    "CFNAI Index",
    "Manufacturing ISM",
    "Manufacturing ISM - Price Paid",
    "Service ISM",
    "Service ISM - Price Paid",
    "US Retail Sales",
]

QUARTERLY_VARS = [
    "US GDP",
    "GDP - Personal Consumption",
    "GDP - Goods Consumption",
    "GDP - Durable Goods Consumption",
    "GDP - Non durable Goods",
    "GDP - Service",
    "GDP - Investment",
]


# =========================================================
# Helpers
# =========================================================
def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def load_raw_csv(path):
    return pd.read_csv(path, header=None)


def build_clean_from_raw_csv(path, selected_vars, sheet_name):
    """
    Structure expected in raw CSV:
    row 0 = old column numbers (0,1,2,...)
    row 1 = Start Date
    row 2 = End Date
    row 3 = readable variable names
    row 4 = Bloomberg tickers
    row 5 = Last Price
    row 6 = Dates / PX_LAST
    row 7+ = actual data
    """
    df_raw = load_raw_csv(path).copy()
    df_raw = df_raw.replace("#N/A N/A", np.nan)
    df_raw = df_raw.replace("#N/A", np.nan)
    df_raw = df_raw.map(clean_text)

    # Fixed structure from your raw csv
    NAMES_ROW = 3
    DATE_MARKER_ROW = 6
    DATA_START_ROW = 7

    # Column names from readable names row
    names = df_raw.iloc[NAMES_ROW].tolist()

    # Find date column from row 'Dates, PX_LAST, ...'
    marker_row = df_raw.iloc[DATE_MARKER_ROW].tolist()
    try:
        date_col_idx = marker_row.index("Dates")
    except ValueError:
        raise ValueError(f"[{sheet_name}] Could not find 'Dates' in row {DATE_MARKER_ROW}")

    columns = []
    for j, val in enumerate(names):
        if j == date_col_idx:
            columns.append("Date")
        else:
            columns.append(val if pd.notna(val) and val != "" else f"Unnamed_{j}")

    # Slice real data
    df = df_raw.iloc[DATA_START_ROW:].copy().reset_index(drop=True)
    df.columns = columns[: df.shape[1]]

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Keep only useful columns that are present
    available = [v for v in selected_vars if v in df.columns]
    missing = [v for v in selected_vars if v not in df.columns]

    print(f"\n[{sheet_name}] Available variables kept: {available}")
    if missing:
        print(f"[{sheet_name}] Missing variables: {missing}")

    df = df[["Date"] + available].copy()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Convert numerics
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"]).copy()

    # Sort
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =========================================================
# Raw file paths
# =========================================================
daily_raw_path = RAW_DIR / "daily_raw.csv"
monthly_raw_path = RAW_DIR / "monthly_raw.csv"
quarterly_raw_path = RAW_DIR / "quarterly_raw.csv"

for p in [daily_raw_path, monthly_raw_path, quarterly_raw_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing raw file: {p}")


# =========================================================
# Clean datasets
# =========================================================
daily_clean = build_clean_from_raw_csv(daily_raw_path, DAILY_VARS, "Daily")
monthly_clean = build_clean_from_raw_csv(monthly_raw_path, MONTHLY_VARS, "Monthly")
quarterly_clean = build_clean_from_raw_csv(quarterly_raw_path, QUARTERLY_VARS, "Quarterly")


# =========================================================
# Save clean datasets
# =========================================================
daily_out = INTERIM_DIR / "daily_clean.csv"
monthly_out = INTERIM_DIR / "monthly_clean.csv"
quarterly_out = INTERIM_DIR / "quarterly_clean.csv"

daily_clean.to_csv(daily_out, index=False)
monthly_clean.to_csv(monthly_out, index=False)
quarterly_clean.to_csv(quarterly_out, index=False)

print("\nClean files created successfully.")
print(f"Saved: {daily_out}")
print(f"Saved: {monthly_out}")
print(f"Saved: {quarterly_out}")

print("\nShapes:")
print("daily_clean    :", daily_clean.shape)
print("monthly_clean  :", monthly_clean.shape)
print("quarterly_clean:", quarterly_clean.shape)

print("\nDaily preview:")
print(daily_clean.head())

print("\nMonthly preview:")
print(monthly_clean.head())

print("\nQuarterly preview:")
print(quarterly_clean.head())