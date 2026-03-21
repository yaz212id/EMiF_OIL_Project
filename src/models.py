import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/interim/daily_clean.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Returns
df['r_oil'] = np.log(df['WTI futures'] / df['WTI futures'].shift(1))
df['r_sp500'] = np.log(df['S&P500'] / df['S&P500'].shift(1))

# Volatility proxy
df['vol_sp500'] = df['r_sp500']**2

# Lag variables
df['r_oil_lag'] = df['r_oil'].shift(1)
df['r_sp500_lag'] = df['r_sp500'].shift(1)
df['vol_lag'] = df['vol_sp500'].shift(1)

# Dummy early 2000s
df['D_2000'] = ((df['Date'].dt.year >= 1997) & (df['Date'].dt.year <= 2006)).astype(int)

# Interaction
df['interaction'] = df['r_oil'] * df['D_2000']

# Drop NA
df = df.dropna()

print(df.describe())