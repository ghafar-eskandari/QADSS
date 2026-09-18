import pandas as pd
import numpy as np

MODEL_READY = r".\data\processed\webmelat_model_ready_v2.csv"
FEATURES = r".\data\processed\webmelat_features_v2.csv"
FLAGS = r".\data\processed\event_aware_data_flags.csv"

OUTPUT_CLEAN = r".\data\processed\webmelat_model_ready_v2_clean.csv"
OUTPUT_IMPACT = r".\data\processed\clean_dataset_impact.csv"


# ============================================================
# STAGE 20.9
# CLEAN DATASET CONSTRUCTION & IMPACT ANALYSIS
# ============================================================

print()
print("=== STAGE 20.9: CLEAN DATASET CONSTRUCTION ===")
print()


# ------------------------------------------------------------
# 1. Load model-ready V2
# ------------------------------------------------------------

model_df = pd.read_csv(
    MODEL_READY,
    dtype={"dEven": str}
)

model_df["dEven"] = (
    model_df["dEven"]
    .astype(str)
    .str.strip()
)

print("Original model-ready rows:", len(model_df))


# ------------------------------------------------------------
# 2. Load full V2 features
# ------------------------------------------------------------

feature_df = pd.read_csv(
    FEATURES,
    dtype={"dEven": str}
)

feature_df["dEven"] = (
    feature_df["dEven"]
    .astype(str)
    .str.strip()
)

print("Original feature rows:", len(feature_df))


# ------------------------------------------------------------
# 3. Load event flags
# ------------------------------------------------------------

flags = pd.read_csv(
    FLAGS,
    dtype={"dEven": str}
)

flags["dEven"] = (
    flags["dEven"]
    .astype(str)
    .str.strip()
)

print("Event flag rows:", len(flags))


# ------------------------------------------------------------
# 4. Create affected-date list
# ------------------------------------------------------------

affected_dates = set(
    flags.loc[
        flags["event_affected"] == 1,
        "dEven"
    ]
)


# ------------------------------------------------------------
# 5. Flag model-ready observations
# ------------------------------------------------------------

model_df["event_affected"] = (
    model_df["dEven"]
    .isin(affected_dates)
    .astype(int)
)


# ------------------------------------------------------------
# 6. Summary before cleaning
# ------------------------------------------------------------

total_rows = len(model_df)

affected_rows = int(
    model_df["event_affected"].sum()
)

clean_rows = total_rows - affected_rows

print()
print("=== MODEL-READY IMPACT ===")
print("Total rows:", total_rows)
print("Event-affected rows:", affected_rows)
print("Clean rows:", clean_rows)

if total_rows > 0:
    print(
        "Affected rate:",
        round(
            affected_rows / total_rows,
            4
        )
    )


# ------------------------------------------------------------
# 7. Construct clean model-ready dataset
# ------------------------------------------------------------

clean_df = model_df[
    model_df["event_affected"] == 0
].copy()


clean_df = clean_df.drop(
    columns=["event_affected"]
)


# ------------------------------------------------------------
# 8. Check target distribution
# ------------------------------------------------------------

print()
print("=== TARGET DISTRIBUTION ===")

print("ORIGINAL:")

print(
    model_df["target"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print("CLEAN:")

print(
    clean_df["target"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 9. Missing / infinite checks
# ------------------------------------------------------------

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
    "no_trade"
]


print()
print("=== CLEAN DATA QUALITY ===")

missing_total = int(
    clean_df[FEATURES_V2]
    .isna()
    .sum()
    .sum()
)

numeric_features = (
    clean_df[FEATURES_V2]
    .select_dtypes(
        include=[np.number]
    )
)

infinite_total = int(
    np.isinf(numeric_features)
    .sum()
    .sum()
)

print(
    "Missing feature values:",
    missing_total
)

print(
    "Infinite feature values:",
    infinite_total
)


# ------------------------------------------------------------
# 10. Extreme feature check
# ------------------------------------------------------------

print()
print("=== CLEAN EXTREME FEATURE CHECK ===")

for feature in FEATURES_V2:

    s = clean_df[feature].dropna()

    if len(s) == 0:
        continue

    print(
        f"{feature:20s} "
        f"min={s.min(): .4f} "
        f"max={s.max(): .4f}"
    )


# ------------------------------------------------------------
# 11. Compare original vs clean
# ------------------------------------------------------------

comparison_rows = []

for feature in FEATURES_V2:

    original = model_df[feature].dropna()
    clean = clean_df[feature].dropna()

    comparison_rows.append(
        {
            "feature": feature,

            "original_n": len(original),
            "clean_n": len(clean),

            "original_min": original.min(),
            "clean_min": clean.min(),

            "original_max": original.max(),
            "clean_max": clean.max(),

            "original_mean": original.mean(),
            "clean_mean": clean.mean(),

            "original_std": original.std(),
            "clean_std": clean.std()
        }
    )


impact_df = pd.DataFrame(
    comparison_rows
)


# ------------------------------------------------------------
# 12. Save clean dataset
# ------------------------------------------------------------

clean_df.to_csv(
    OUTPUT_CLEAN,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 13. Save impact report
# ------------------------------------------------------------

impact_df.to_csv(
    OUTPUT_IMPACT,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

print()
print("=== STAGE 20.9 SUMMARY ===")

print(
    "Original model-ready:",
    total_rows
)

print(
    "Event-affected:",
    affected_rows
)

print(
    "Clean model-ready:",
    clean_rows
)

print()
print(
    "Saved clean dataset:",
    OUTPUT_CLEAN
)

print(
    "Saved impact report:",
    OUTPUT_IMPACT
)

print()
print("IMPORTANT:")
print("No model was trained in Stage 20.9.")
print("The clean dataset is prepared for the next validation stage.")