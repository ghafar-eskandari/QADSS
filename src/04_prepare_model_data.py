import pandas as pd

INPUT_PATH = "data/processed/webmelat_model_data.csv"
OUTPUT_PATH = "data/processed/webmelat_model_ready.csv"

FEATURES = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_20",
    "price_to_ma20",
    "volatility_20",
    "volume",
    "no_trade",
]

TARGET = "target"


# Load model dataset
df = pd.read_csv(INPUT_PATH)

# Ensure chronological order
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep only rows with valid features and target
model_df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

# Convert target to integer
model_df[TARGET] = model_df[TARGET].astype(int)

# Save model-ready dataset
model_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Model-ready dataset created successfully.")
print("Original rows:", len(df))
print("Model-ready rows:", len(model_df))
print("Rows removed:", len(df) - len(model_df))
print("Features:", len(FEATURES))
print("Target distribution:")
print(model_df[TARGET].value_counts().sort_index())
print("Output:", OUTPUT_PATH)