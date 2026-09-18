import pandas as pd
import numpy as np

INPUT = r".\data\processed\corporate_action_window_match.csv"
OUTPUT = r".\data\processed\corporate_action_residual_check.csv"

df = pd.read_csv(INPUT)

df["adjusted_before"] = (
    df["close_before"] * df["adjustment_factor"]
)

df["adjusted_return"] = (
    df["close_after"] / df["adjusted_before"] - 1
)

df["raw_abs_return"] = df["raw_return"].abs()
df["adjusted_abs_return"] = df["adjusted_return"].abs()

df["explanation_ratio"] = np.where(
    df["raw_abs_return"] > 0,
    1 - (
        df["adjusted_abs_return"] / df["raw_abs_return"]
    ),
    np.nan
)

def classify(row):
    if pd.isna(row["adjustment_factor"]):
        return "NO_OFFICIAL_ADJUSTMENT"
    
    if row["raw_abs_return"] < 0.05:
        return "SMALL_RAW_MOVE"
    
    if row["adjusted_abs_return"] <= 0.05:
        return "EXPLAINED_BY_ADJUSTMENT"
    
    return "UNEXPLAINED_AFTER_ADJUSTMENT"

df["residual_status"] = df.apply(classify, axis=1)

columns = [
    "event_date",
    "close_before",
    "close_after",
    "raw_return",
    "adjustment_date",
    "tsetmc_not_adjusted_close",
    "tsetmc_adjusted_close",
    "adjustment_factor",
    "adjusted_before",
    "adjusted_return",
    "explanation_ratio",
    "days_from_event",
    "residual_status",
]

df[columns].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(df[columns].to_string(index=False))

print()
print("=== RESIDUAL STATUS COUNTS ===")
print(df["residual_status"].value_counts(dropna=False))

print()
print("Saved:", OUTPUT)
