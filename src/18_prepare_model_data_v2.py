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
    / "webmelat_features_v2.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_model_ready_v2.csv"
)


# --------------------------------------------------
# 2. V2 feature set
# --------------------------------------------------

FEATURES_V2 = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "close_to_ma5",
    "close_to_ma20",
    "ma5_to_ma20",
    "ma5_slope_5d",
    "ma20_slope_5d",
    "volatility_20",
    "volume_ratio_20",
    "no_trade",
]


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 4. Create target
# --------------------------------------------------

df["future_return_5d"] = (
    df["close"].shift(-5)
    / df["close"]
    - 1
)

df["target"] = (
    df["future_return_5d"] > 0
).astype("Int64")

df.loc[
    df["future_return_5d"].isna(),
    "target"
] = pd.NA


# --------------------------------------------------
# 5. Check required columns
# --------------------------------------------------

required_columns = FEATURES_V2 + ["target"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# --------------------------------------------------
# 6. Prepare model data
# --------------------------------------------------
model_df = df[
    ["dEven"]
    + FEATURES_V2
    + ["target"]
].copy()

# --------------------------------------------------
# 7. Remove missing values
# --------------------------------------------------

before_rows = len(model_df)

model_df = model_df.dropna(
    subset=FEATURES_V2 + ["target"]
).copy()

after_rows = len(model_df)

removed_missing = (
    before_rows - after_rows
)


# --------------------------------------------------
# 8. Remove infinite values
# --------------------------------------------------

model_df = model_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

before_inf_drop = len(model_df)

model_df = model_df.dropna(
    subset=FEATURES_V2 + ["target"]
).copy()

removed_infinite = (
    before_inf_drop - len(model_df)
)


# --------------------------------------------------
# 9. Convert target to integer
# --------------------------------------------------

model_df["target"] = (
    model_df["target"]
    .astype(int)
)


# --------------------------------------------------
# 10. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 11. Report
# --------------------------------------------------

print("=" * 80)
print("MODEL READY V2")
print("=" * 80)

print(
    f"Original rows: {before_rows}"
)

print(
    f"Final rows: {len(model_df)}"
)

print(
    f"Removed due to missing values: "
    f"{removed_missing}"
)

print(
    f"Removed due to infinite values: "
    f"{removed_infinite}"
)

print(
    f"Number of features: "
    f"{len(FEATURES_V2)}"
)


# --------------------------------------------------
# 12. Feature list
# --------------------------------------------------

print("\n")
print("=" * 80)
print("FEATURE SET V2")
print("=" * 80)

for i, feature in enumerate(
    FEATURES_V2,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# --------------------------------------------------
# 13. Target distribution
# --------------------------------------------------

print("\n")
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(
    model_df["target"]
    .value_counts()
    .sort_index()
    .to_string()
)


# --------------------------------------------------
# 14. Final validation
# --------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL CHECK")
print("=" * 80)

print(
    f"Missing values: "
    f"{model_df.isna().sum().sum()}"
)

numeric_df = model_df.select_dtypes(
    include="number"
)

infinite_count = numeric_df.isin(
    [float("inf"), float("-inf")]
).sum().sum()

print(
    f"Infinite values: {infinite_count}"
)

print("\n")
print(
    f"Saved to:\n{OUTPUT_PATH}"
)

print("=" * 80)