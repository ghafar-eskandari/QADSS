import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

FILE = BASE / "data" / "processed" / "webmelat_model_ready_v2.csv"

FEATURES = [
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
    "no_trade"
]

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(
    df["dEven"],
    errors="coerce"
)

print()
print("=" * 65)
print("STAGE 21: FINAL DATASET VALIDATION")
print("=" * 65)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print(
    f"Date: {df['date'].min().date()} "
    f"-> {df['date'].max().date()}"
)

print("\n--- Target ---")

print(
    df["target"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n--- Missing values ---")

missing = df[FEATURES + ["target"]].isna().sum()

print(
    missing[missing > 0]
    if missing.sum() > 0
    else "NONE"
)

print("\n--- Infinite values ---")

numeric = df[FEATURES].select_dtypes(
    include=[np.number]
)

inf_count = np.isinf(
    numeric.to_numpy()
).sum()

print(f"Inf values: {inf_count}")

print("\n--- Duplicate dates ---")

duplicates = df["date"].duplicated().sum()

print(f"Duplicate dates: {duplicates}")

print("\n--- Target validity ---")

valid_targets = set(
    df["target"].dropna().unique()
)

print(
    f"Target values: {sorted(valid_targets)}"
)

print("\n--- Feature count ---")

print(
    f"Expected features: {len(FEATURES)}"
)

print(
    f"Available features: "
    f"{sum(f in df.columns for f in FEATURES)}"
)

print("\n--- Final validation ---")

checks = {
    "Rows > 0":
        len(df) > 0,

    "All features exist":
        all(f in df.columns for f in FEATURES),

    "No missing feature/target":
        missing.sum() == 0,

    "No infinite values":
        inf_count == 0,

    "No duplicate dates":
        duplicates == 0,

    "Target only 0/1":
        valid_targets.issubset({0, 1}),

    "Chronological dates":
        df["date"].is_monotonic_increasing
}

for name, result in checks.items():

    print(
        f"[{'PASS' if result else 'FAIL'}] "
        f"{name}"
    )

print()

if all(checks.values()):

    print(
        "FINAL VALIDATION: PASS"
    )

    print(
        "Dataset is ready for Stage 22."
    )

else:

    print(
        "FINAL VALIDATION: FAIL"
    )

    print(
        "Dataset requires correction before Stage 22."
    )