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


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 3. Identify suspicious ratio values
# --------------------------------------------------

ratio_features = [
    "volume_ratio_20",
    "volatility_ratio_60",
]


# --------------------------------------------------
# 4. Basic status
# --------------------------------------------------

print("=" * 80)
print("RATIO BEHAVIOR CHECK")
print("=" * 80)

print(f"Total rows: {len(df)}")


# --------------------------------------------------
# 5. Analyze each ratio
# --------------------------------------------------

for feature in ratio_features:

    print("\n")
    print("=" * 80)
    print(f"FEATURE: {feature}")
    print("=" * 80)

    series = df[feature]

    print(
        f"Missing values: "
        f"{series.isna().sum()}"
    )

    print(
        f"Zero values: "
        f"{(series == 0).sum()}"
    )

    print(
        f"> 5 values: "
        f"{(series > 5).sum()}"
    )

    print(
        f"> 10 values: "
        f"{(series > 10).sum()}"
    )


# --------------------------------------------------
# 6. Focus on no-trade relationship
# --------------------------------------------------

print("\n")
print("=" * 80)
print("RELATIONSHIP WITH NO-TRADE")
print("=" * 80)

for feature in ratio_features:

    print("\n")
    print(f"FEATURE: {feature}")

    grouped = (
        df.groupby("no_trade")[feature]
        .agg(
            observations="count",
            missing="size",
            mean="mean",
            median="median",
            maximum="max",
            zeros=lambda x: (x == 0).sum(),
            greater_than_5=lambda x: (x > 5).sum(),
            greater_than_10=lambda x: (x > 10).sum(),
        )
    )

    print(
        grouped.to_string()
    )


# --------------------------------------------------
# 7. Inspect extreme observations
# --------------------------------------------------

print("\n")
print("=" * 80)
print("EXTREME OBSERVATIONS")
print("=" * 80)

for feature in ratio_features:

    print("\n")
    print(f"TOP VALUES: {feature}")

    columns = [
        "dEven",
        "close",
        "volume",
        "no_trade",
        "volatility_20",
        feature,
    ]

    available_columns = [
        c for c in columns
        if c in df.columns
    ]

    top_rows = (
        df[available_columns]
        .sort_values(
            by=feature,
            ascending=False,
            na_position="last"
        )
        .head(15)
    )

    print(
        top_rows.to_string(index=False)
    )


# --------------------------------------------------
# 8. Final conclusion
# --------------------------------------------------

print("\n")
print("=" * 80)
print("CHECK COMPLETED")
print("=" * 80)

print(
    "No data has been modified."
)

print(
    "This script only investigates "
    "the behavior of the ratio features."
)

print("=" * 80)