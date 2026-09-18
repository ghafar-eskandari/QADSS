import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STAGE 20.13
# FINAL EVENT FLAGGING
# ============================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE
    / "data"
    / "processed"
    / "extreme_corporate_action_reconciliation.csv"
)

OUTPUT_FILE = (
    BASE
    / "data"
    / "processed"
    / "final_event_flags.csv"
)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print()
print("=== STAGE 20.13: FINAL EVENT FLAGGING ===")
print()

print(f"Input rows: {len(df)}")


# ------------------------------------------------------------
# 2. Convert numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "raw_return_1d",
    "no_trade_before",
    "no_trade_after",
    "adjustment_factor",
    "adjustment_distance_days",
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# 3. Extreme return flag
# ------------------------------------------------------------
#
# We use absolute daily return >= 20%
# as the extreme-return threshold.
#
# This is an audit threshold, NOT a trading threshold.
# ------------------------------------------------------------

df["extreme_return_flag"] = (
    df["raw_return_1d"].abs() >= 0.20
).astype(int)


# ------------------------------------------------------------
# 4. Reopening flag
# ------------------------------------------------------------
#
# At least 3 no-trade observations immediately before
# OR after the event.
# ------------------------------------------------------------

df["reopening_flag"] = (
    (
        df["no_trade_before"] >= 3
    )
    |
    (
        df["no_trade_after"] >= 3
    )
).astype(int)


# ------------------------------------------------------------
# 5. Adjustment-supported flag
# ------------------------------------------------------------
#
# Only classify as adjustment-supported when the previous
# audit explicitly identified the event as explained by
# an official adjustment AND the adjustment is close enough.
# ------------------------------------------------------------

df["adjustment_supported_flag"] = (
    (
        df["adjustment_status"]
        == "EXPLAINED_BY_ADJUSTMENT"
    )
    &
    (
        df["adjustment_distance_days"] <= 30
    )
).astype(int)


# ------------------------------------------------------------
# 6. Unresolved event flag
# ------------------------------------------------------------
#
# An event is unresolved when:
#
# - it has an extreme return,
# - and there is no confirmed adjustment explanation.
#
# This includes reopening-related events whose price jump
# is not explained by the official adjustment.
# ------------------------------------------------------------

df["unresolved_event_flag"] = (
    (
        df["extreme_return_flag"] == 1
    )
    &
    (
        df["adjustment_supported_flag"] == 0
    )
).astype(int)


# ------------------------------------------------------------
# 7. Structural break flag
# ------------------------------------------------------------
#
# Candidate structural break:
#
# A) very large daily return >= 50%
# AND
# B) either reopening-related
# OR no official adjustment explanation.
#
# This is deliberately a CANDIDATE flag.
# It is NOT a statement that a corporate action occurred.
# ------------------------------------------------------------

df["structural_break_flag"] = (
    (
        df["raw_return_1d"].abs() >= 0.50
    )
    &
    (
        (
            df["reopening_flag"] == 1
        )
        |
        (
            df["adjustment_supported_flag"] == 0
        )
    )
).astype(int)


# ------------------------------------------------------------
# 8. Event affected flag
# ------------------------------------------------------------
#
# Any unresolved extreme or reopening event is considered
# event-affected.
# ------------------------------------------------------------

df["event_affected_flag"] = (
    (
        df["extreme_return_flag"] == 1
    )
    |
    (
        df["reopening_flag"] == 1
    )
).astype(int)


# ------------------------------------------------------------
# 9. Final event category
# ------------------------------------------------------------
#
# This category is only for convenient reporting.
# The individual flags above remain the authoritative fields.
# ------------------------------------------------------------

def classify_event(row):

    if row["adjustment_supported_flag"] == 1:
        return "ADJUSTMENT_SUPPORTED"

    if row["structural_break_flag"] == 1:
        return "STRUCTURAL_BREAK_CANDIDATE"

    if (
        row["reopening_flag"] == 1
        and row["unresolved_event_flag"] == 1
    ):
        return "REOPENING_UNRESOLVED"

    if row["unresolved_event_flag"] == 1:
        return "EXTREME_UNRESOLVED"

    if row["reopening_flag"] == 1:
        return "REOPENING"

    return "NORMAL"


df["final_event_category"] = (
    df.apply(
        classify_event,
        axis=1
    )
)


# ------------------------------------------------------------
# 10. Display flag summary
# ------------------------------------------------------------

print()
print("=== FLAG SUMMARY ===")

flag_columns = [
    "extreme_return_flag",
    "reopening_flag",
    "adjustment_supported_flag",
    "unresolved_event_flag",
    "structural_break_flag",
    "event_affected_flag",
]

for col in flag_columns:

    print(
        f"{col}: "
        f"{df[col].sum()}"
    )


# ------------------------------------------------------------
# 11. Category summary
# ------------------------------------------------------------

print()
print("=== FINAL EVENT CATEGORY ===")

print(
    df[
        "final_event_category"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 12. Show structural-break candidates
# ------------------------------------------------------------

print()
print(
    "=== STRUCTURAL BREAK CANDIDATES ==="
)

structural = df[
    df["structural_break_flag"] == 1
]

if len(structural) > 0:

    print(
        structural[
            [
                "event_date",
                "raw_return_1d",
                "no_trade_before",
                "no_trade_after",
                "adjustment_status",
                "extreme_return_flag",
                "reopening_flag",
                "unresolved_event_flag",
                "structural_break_flag",
                "final_event_category",
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ------------------------------------------------------------
# 13. Show unresolved extreme events
# ------------------------------------------------------------

print()
print(
    "=== UNRESOLVED EXTREME EVENTS ==="
)

unresolved = df[
    df["unresolved_event_flag"] == 1
]

if len(unresolved) > 0:

    print(
        unresolved[
            [
                "event_date",
                "raw_return_1d",
                "reopening_flag",
                "adjustment_status",
                "final_event_category",
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ------------------------------------------------------------
# 14. Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 15. Final summary
# ------------------------------------------------------------

print()
print("=== STAGE 20.13 SUMMARY ===")
print()

print(
    f"Total extreme events: {len(df)}"
)

print(
    f"Extreme return events: "
    f"{df['extreme_return_flag'].sum()}"
)

print(
    f"Reopening-related events: "
    f"{df['reopening_flag'].sum()}"
)

print(
    f"Adjustment-supported events: "
    f"{df['adjustment_supported_flag'].sum()}"
)

print(
    f"Unresolved events: "
    f"{df['unresolved_event_flag'].sum()}"
)

print(
    f"Structural-break candidates: "
    f"{df['structural_break_flag'].sum()}"
)

print(
    f"Event-affected events: "
    f"{df['event_affected_flag'].sum()}"
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("IMPORTANT:")
print("No rows were deleted.")
print("No model was trained.")
print(
    "Individual event flags are authoritative; "
    "final_event_category is for reporting only."
)