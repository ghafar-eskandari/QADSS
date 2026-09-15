import pandas as pd

INPUT_PATH = "data/processed/webmelat_features.csv"
OUTPUT_PATH = "data/processed/webmelat_model_data.csv"

# Load feature dataset
df = pd.read_csv(INPUT_PATH)

# Ensure chronological order
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Future 5-trading-day return
df["future_return_5d"] = df["close"].shift(-5) / df["close"] - 1

# Binary target:
# 1 = positive return over the next 5 trading days
# 0 = zero or negative return
df["target"] = (df["future_return_5d"] > 0).astype("Int64")

# The last 5 observations have no future 5-day return
# and therefore no valid target.
df.loc[df["future_return_5d"].isna(), "target"] = pd.NA

# Save model-ready dataset
df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Target dataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Valid targets:", df["target"].notna().sum())
print("Target = 1:", (df["target"] == 1).sum())
print("Target = 0:", (df["target"] == 0).sum())
print("Missing targets:", df["target"].isna().sum())
print("Output:", OUTPUT_PATH)