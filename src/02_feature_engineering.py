import pandas as pd

INPUT_PATH = "data/processed/webmelat_clean.csv"
OUTPUT_PATH = "data/processed/webmelat_features.csv"


# Load processed data
df = pd.read_csv(INPUT_PATH)

# Ensure date is datetime
df["date"] = pd.to_datetime(df["date"])


# =========================
# Basic Price Features
# =========================

# Daily return
df["return_1d"] = df["close"].pct_change()

# 5-day return
df["return_5d"] = df["close"].pct_change(5)

# 5-day moving average
df["ma_5"] = df["close"].rolling(window=5).mean()

# 20-day moving average
df["ma_20"] = df["close"].rolling(window=20).mean()

# Distance from 20-day moving average
df["price_to_ma20"] = df["close"] / df["ma_20"] - 1

# 20-day rolling volatility
df["volatility_20"] = df["return_1d"].rolling(window=20).std()

# Trading volume
df["volume"] = df["zTotTran"]


# =========================
# Trading Status
# =========================

# 1 = no trade, 0 = traded
df["no_trade"] = df["no_trade"].astype(int)


# Save feature dataset
df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("Feature dataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Output:", OUTPUT_PATH)