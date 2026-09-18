import pandas as pd
import numpy as np

CLEAN_DATA = r".\data\processed\webmelat_model_ready_v2_clean.csv"
FLAGS = r".\data\processed\event_aware_data_flags.csv"

OUTPUT = r".\data\processed\clean_data_audit.csv"

print()
print("=== STAGE 20.10: CLEAN DATA AUDIT ===")
print()

# ---------------------------------------------------------
# 1. Load clean dataset
# ---------------------------------------------------------

df = pd.read_csv(
    CLEAN_DATA,
    dtype={"dEven": str}
)

df["dEven"] = df["dEven"].astype(str).str.strip()

print("Clean dataset rows:", len(df))
print(
    "Date range:",
    df["dEven"].min(),
    "to",
    df["dEven"].max()
)

# ---------------------------------------------------------
# 2. Load event flags
# ---------------------------------------------------------

flags = pd.read_csv(
    FLAGS,
    dtype={"dEven": str}
)

flags["dEven"] = flags["dEven"].astype(str).str.strip()

event_dates = set(
    flags.loc[
        flags["event_affected"] == 1,
        "dEven"
    ]
)

# ---------------------------------------------------------
# 3. Define extreme-return thresholds
# ---------------------------------------------------------

THRESHOLDS = {
    "return_1d": 0.20,
    "return_5d": 0.30,
    "return_10d": 0.40,
    "return_20d": 0.50
}

# ---------------------------------------------------------
# 4. Detect extreme observations
# ---------------------------------------------------------

audit_rows = []

for feature, threshold in THRESHOLDS.items():

    mask = df[feature].abs() >= threshold

    extreme = df.loc[
        mask,
        ["dEven", feature]
    ].copy()

    for _, row in extreme.iterrows():

        date = str(row["dEven"])

        audit_rows.append(
            {
                "dEven": date,
                "feature": feature,
                "value": row[feature],
                "abs_value": abs(row[feature]),
                "threshold": threshold,
                "event_flag": int(date in event_dates)
            }
        )

audit_df = pd.DataFrame(audit_rows)

# ---------------------------------------------------------
# 5. Sort
# ---------------------------------------------------------

if len(audit_df) > 0:
    audit_df = audit_df.sort_values(
        ["dEven", "feature"]
    ).reset_index(drop=True)

# ---------------------------------------------------------
# 6. Print summary
# ---------------------------------------------------------

print()
print("=== EXTREME VALUES REMAINING IN CLEAN DATA ===")

if len(audit_df) == 0:

    print("No extreme values detected.")

else:

    print(
        audit_df[
            [
                "dEven",
                "feature",
                "value",
                "threshold",
                "event_flag"
            ]
        ].to_string(index=False)
    )

# ---------------------------------------------------------
# 7. Summary by feature
# ---------------------------------------------------------

print()
print("=== EXTREME VALUE COUNT BY FEATURE ===")

if len(audit_df) > 0:

    print(
        audit_df
        .groupby("feature")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

# ---------------------------------------------------------
# 8. Event association
# ---------------------------------------------------------

print()
print("=== EVENT ASSOCIATION ===")

if len(audit_df) > 0:

    event_counts = (
        audit_df["event_flag"]
        .value_counts()
        .sort_index()
    )

    print(event_counts.to_string())

    event_rate = audit_df["event_flag"].mean()

    print(
        "Extreme observations associated with event flags:",
        round(event_rate, 4)
    )

# ---------------------------------------------------------
# 9. Most extreme observations
# ---------------------------------------------------------

print()
print("=== TOP EXTREME OBSERVATIONS ===")

if len(audit_df) > 0:

    print(
        audit_df
        .sort_values(
            "abs_value",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

# ---------------------------------------------------------
# 10. Save audit
# ---------------------------------------------------------

audit_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== STAGE 20.10 SUMMARY ===")
print()

print(
    "Extreme observations detected:",
    len(audit_df)
)

print(
    "Saved audit:",
    OUTPUT
)

print()
print("IMPORTANT:")
print("No rows were deleted.")
print("No model was trained.")
print("This stage only identifies remaining extreme observations.")