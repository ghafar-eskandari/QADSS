from pathlib import Path

import numpy as np
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


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 3. New features
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Basic information
# --------------------------------------------------

print("=" * 80)
print("NORMALIZED FEATURE VALIDATION")
print("=" * 80)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# 5. Missing values
# --------------------------------------------------

print("\n")
print("=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = df[NEW_FEATURES].isna().sum()

print(missing.to_string())


# --------------------------------------------------
# 6. Infinite values
# --------------------------------------------------

print("\n")
print("=" * 80)
print("INFINITE VALUES")
print("=" * 80)

numeric_df = df[NEW_FEATURES].select_dtypes(
    include=[np.number]
)

infinite_counts = np.isinf(
    numeric_df
).sum()

print(infinite_counts.to_string())


# --------------------------------------------------
# 7. Summary statistics
# --------------------------------------------------

print("\n")
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

summary = df[NEW_FEATURES].describe().T

print(
    summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ].to_string()
)


# --------------------------------------------------
# 8. Extreme values
# --------------------------------------------------

print("\n")
print("=" * 80)
print("EXTREME VALUE CHECK")
print("=" * 80)

for feature in NEW_FEATURES:

    series = df[feature].dropna()

    if len(series) == 0:
        print(
            f"{feature}: no valid observations"
        )
        continue

    q01 = series.quantile(0.01)
    q99 = series.quantile(0.99)

    extreme_count = (
        (series < q01) |
        (series > q99)
    ).sum()

    print(
        f"{feature:25s} "
        f"1%={q01: .6f} "
        f"99%={q99: .6f} "
        f"extreme={extreme_count}"
    )


# --------------------------------------------------
# 9. Final status
# --------------------------------------------------

total_missing = (
    missing.sum()
)

total_infinite = (
    infinite_counts.sum()
)

print("\n")
print("=" * 80)
print("VALIDATION STATUS")
print("=" * 80)

print(
    f"Total missing values: "
    f"{total_missing}"
)

print(
    f"Total infinite values: "
    f"{total_infinite}"
)

if (
    total_missing == 0
    and total_infinite == 0
):
    print(
        "STATUS: PASS"
    )
else:
    print(
        "STATUS: REVIEW REQUIRED"
    )

print("=" * 80)