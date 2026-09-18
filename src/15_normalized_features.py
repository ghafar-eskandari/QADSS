from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_features_v2.csv"
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["dEven"] = pd.to_datetime(
    df["dEven"].astype(str),
    format="%Y%m%d"
)


# --------------------------------------------------
# 3. Normalized price features
# --------------------------------------------------

# Distance of current price from moving averages
df["close_to_ma5"] = (
    df["close"] / df["ma_5"] - 1
)

df["close_to_ma20"] = (
    df["close"] / df["ma_20"] - 1
)


# --------------------------------------------------
# 4. Moving-average relationship
# --------------------------------------------------

# Short-term trend relative to medium-term trend
df["ma5_to_ma20"] = (
    df["ma_5"] / df["ma_20"] - 1
)


# --------------------------------------------------
# 5. Moving-average slopes
# --------------------------------------------------

df["ma5_slope_5d"] = (
    df["ma_5"] / df["ma_5"].shift(5) - 1
)

df["ma20_slope_5d"] = (
    df["ma_20"] / df["ma_20"].shift(5) - 1
)


# --------------------------------------------------
# 6. Price momentum features
# --------------------------------------------------

df["return_10d"] = (
    df["close"].pct_change(10)
)

df["return_20d"] = (
    df["close"].pct_change(20)
)


# --------------------------------------------------
# 7. Volume normalization
# --------------------------------------------------

df["volume_ma20"] = (
    df["volume"]
    .rolling(window=20)
    .mean()
)

df["volume_ratio_20"] = (
    df["volume"] / df["volume_ma20"]
)


# --------------------------------------------------
# 8. Volatility regime
# --------------------------------------------------

df["volatility_20_ma60"] = (
    df["volatility_20"]
    .rolling(window=60)
    .mean()
)

df["volatility_ratio_60"] = (
    df["volatility_20"]
    / df["volatility_20_ma60"]
)


# --------------------------------------------------
# 9. Remove infinite values
# --------------------------------------------------

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# --------------------------------------------------
# 10. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 11. Report
# --------------------------------------------------

print("=" * 70)
print("NORMALIZED FEATURE ENGINEERING - VERSION 2")
print("=" * 70)

print(
    f"Input rows: {len(df)}"
)

print(
    f"Output columns: {len(df.columns)}"
)

print("\nNew features:")

NEW_FEATURES = [
    "close_to_ma5",
    "close_to_ma20",
    "ma5_to_ma20",
    "ma5_slope_5d",
    "ma20_slope_5d",
    "return_10d",
    "return_20d",
    "volume_ma20",
    "volume_ratio_20",
    "volatility_20_ma60",
    "volatility_ratio_60",
]

for feature in NEW_FEATURES:
    print(
        f"  - {feature}"
    )

print("\n")
print(
    f"Saved to:\n{OUTPUT_PATH}"
)

print("=" * 70)