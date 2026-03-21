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
# Variables kept for the project
# =========================================================
# WTI removed on purpose; Brent kept as main oil benchmark
DAILY_VARS = [
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


def parse_dates_strict(series: pd.Series) -> pd.Series:
    """
    Parse dates safely from raw csv.
    Main expected format: YYYY-MM-DD HH:MM:SS
    Fallback: YYYY-MM-DD
    """
    s = series.astype(str).str.strip()

    dt = pd.to_datetime(
        s,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(
            s[mask],
            format="%Y-%m-%d",
            errors="coerce"
        )
        dt.loc[mask] = dt2

    return dt


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df = df.replace("#N/A N/A", np.nan)
    df = df.replace("#N/A", np.nan)
    df = df.map(clean_text)
    return df


def build_clean_from_raw_csv(path: Path, selected_vars: list[str], sheet_name: str) -> pd.DataFrame:
    """
    Expected raw csv structure:
    row 0 = old integer column names
    row 1 = Start Date
    row 2 = End Date
    row 3 = readable variable names
    row 4 = Bloomberg tickers
    row 5 = Last Price
    row 6 = Dates / PX_LAST
    row 7+ = actual data
    """
    df_raw = load_raw_csv(path)

    NAMES_ROW = 3
    MARKER_ROW = 6
    DATA_START_ROW = 7

    names = df_raw.iloc[NAMES_ROW].tolist()
    markers = df_raw.iloc[MARKER_ROW].tolist()

    if "Dates" not in markers:
        raise ValueError(f"[{sheet_name}] 'Dates' not found in marker row.")

    date_col_idx = markers.index("Dates")

    columns = []
    for j, val in enumerate(names):
        if j == date_col_idx:
            columns.append("Date")
        else:
            if pd.isna(val) or val == "":
                columns.append(f"Unnamed_{j}")
            else:
                columns.append(val)

    df = df_raw.iloc[DATA_START_ROW:].copy().reset_index(drop=True)
    df.columns = columns[: df.shape[1]]

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Rename one variable for consistency
    rename_map = {
        "High yield index yield to worst": "High yield index",
    }
    df = df.rename(columns=rename_map)

    available = [v for v in selected_vars if v in df.columns]
    missing = [v for v in selected_vars if v not in df.columns]

    print(f"\n[{sheet_name}] Variables kept: {available}")
    if missing:
        print(f"[{sheet_name}] Missing variables: {missing}")

    df = df[["Date"] + available].copy()

    # Critical fix: explicit date parsing, no dayfirst guessing
    df["Date"] = parse_dates_strict(df["Date"])

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"]).copy()
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
# Build clean datasets
# =========================================================
daily_clean = build_clean_from_raw_csv(daily_raw_path, DAILY_VARS, "Daily")
monthly_clean = build_clean_from_raw_csv(monthly_raw_path, MONTHLY_VARS, "Monthly")
quarterly_clean = build_clean_from_raw_csv(quarterly_raw_path, QUARTERLY_VARS, "Quarterly")


# =========================================================
# Save
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