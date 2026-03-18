import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "Data" / "raw"

print("SCRIPT STARTED")
print(f"BASE_DIR: {BASE_DIR}")
print(f"RAW_DIR exists: {RAW_DIR.exists()}")
print(f"RAW_DIR contents: {[p.name for p in RAW_DIR.iterdir()]}")

xlsx_files = list(RAW_DIR.glob("*.xlsx"))

if not xlsx_files:
    raise FileNotFoundError(f"No .xlsx file found in {RAW_DIR}")

file_path = xlsx_files[0]

print(f"Reading file: {file_path}")
print(f"File exists: {file_path.exists()}")

daily = pd.read_excel(file_path, sheet_name="Daily", header=None)
monthly = pd.read_excel(file_path, sheet_name="Monthly", header=None)
quarterly = pd.read_excel(file_path, sheet_name="Quarterly", header=None)

daily.to_csv(RAW_DIR / "daily_raw.csv", index=False)
monthly.to_csv(RAW_DIR / "monthly_raw.csv", index=False)
quarterly.to_csv(RAW_DIR / "quarterly_raw.csv", index=False)

print("CSV files created successfully.")