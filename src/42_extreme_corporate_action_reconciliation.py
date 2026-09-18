import json
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 20.12
# Extreme Event ↔ Corporate Action Reconciliation
# ============================================================

BASE = Path(__file__).resolve().parents[1]

EXTREME_FILE = (
    BASE / "data" / "processed" /
    "extreme_event_classification.csv"
)

ADJUSTMENT_FILE = (
    BASE / "data" / "processed" /
    "corporate_action_residual_check.csv"
)

RAW_FILE = (
    BASE / "data" / "raw" /
    "webmelat_tsetmc_raw.json"
)

OUTPUT_FILE = (
    BASE / "data" / "processed" /
    "extreme_corporate_action_reconciliation.csv"
)


# ------------------------------------------------------------
# 1. Load files
# ------------------------------------------------------------

extreme = pd.read_csv(EXTREME_FILE)
adjustment = pd.read_csv(ADJUSTMENT_FILE)

with open(RAW_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

raw_df = pd.DataFrame(raw["closingPriceDaily"])


# ------------------------------------------------------------
# 2. Normalize dates
# ------------------------------------------------------------

def normalize_date(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    if "-" in value:
        dt = pd.to_datetime(value, errors="coerce")
    else:
        value = value.replace(".0", "")

        if len(value) == 8:
            dt = pd.to_datetime(
                value,
                format="%Y%m%d",
                errors="coerce"
            )
        else:
            dt = pd.to_datetime(value, errors="coerce")

    if pd.isna(dt):
        return np.nan

    return dt.strftime("%Y-%m-%d")


# Extreme dates
extreme["date_key"] = extreme["dEven"].apply(normalize_date)

# Adjustment dates
if "adjustment_date" in adjustment.columns:
    adjustment["date_key"] = (
        adjustment["adjustment_date"]
        .apply(normalize_date)
    )

elif "dEven" in adjustment.columns:
    adjustment["date_key"] = (
        adjustment["dEven"]
        .apply(normalize_date)
    )

else:
    raise ValueError(
        "Adjustment file does not contain adjustment_date or dEven."
    )


# ------------------------------------------------------------
# 3. Prepare raw TSETMC data
# ------------------------------------------------------------

raw_df["pClosing"] = pd.to_numeric(
    raw_df["pClosing"],
    errors="coerce"
)

raw_df["zTotTran"] = pd.to_numeric(
    raw_df["zTotTran"],
    errors="coerce"
)

raw_df["date_key"] = raw_df["dEven"].apply(normalize_date)

raw_df = (
    raw_df
    .sort_values("date_key")
    .reset_index(drop=True)
)

raw_df["no_trade"] = (
    raw_df["zTotTran"] == 0
).astype(int)

raw_df["previous_close"] = (
    raw_df["pClosing"].shift(1)
)

raw_df["raw_return_1d"] = (
    raw_df["pClosing"] /
    raw_df["previous_close"] - 1
)


# ------------------------------------------------------------
# 4. Prepare adjustment data
# ------------------------------------------------------------

# The previous version incorrectly assumed that
# corporate_action_residual_check.csv contained raw_return_1d.
# We do NOT depend on that column anymore.

if "adjustment_factor" in adjustment.columns:
    adjustment["adjustment_factor"] = pd.to_numeric(
        adjustment["adjustment_factor"],
        errors="coerce"
    )
else:
    adjustment["adjustment_factor"] = np.nan


if "days_difference" in adjustment.columns:
    adjustment["days_difference"] = pd.to_numeric(
        adjustment["days_difference"],
        errors="coerce"
    )
else:
    adjustment["days_difference"] = np.nan


if "status" in adjustment.columns:
    adjustment_status_column = "status"

elif "residual_status" in adjustment.columns:
    adjustment_status_column = "residual_status"

elif "adjustment_status" in adjustment.columns:
    adjustment_status_column = "adjustment_status"

else:
    adjustment["status"] = "UNKNOWN"
    adjustment_status_column = "status"


# ------------------------------------------------------------
# 5. Reconcile each extreme event
# ------------------------------------------------------------

results = []

for _, event in extreme.iterrows():

    event_date = pd.to_datetime(
        event["date_key"],
        errors="coerce"
    )

    if pd.isna(event_date):
        continue

    event_raw_return = pd.to_numeric(
        event.get("raw_return_1d", np.nan),
        errors="coerce"
    )

    if pd.isna(event_raw_return):

        raw_match = raw_df[
            raw_df["date_key"] == event["date_key"]
        ]

        if len(raw_match) > 0:
            event_raw_return = raw_match.iloc[0][
                "raw_return_1d"
            ]

    event_raw_return = float(event_raw_return)

    event_no_trade_before = int(
        pd.to_numeric(
            event.get("no_trade_before", 0),
            errors="coerce"
        )
    )

    event_no_trade_after = int(
        pd.to_numeric(
            event.get("no_trade_after", 0),
            errors="coerce"
        )
    )


    # --------------------------------------------------------
    # Find adjustment records within ±60 calendar days
    # --------------------------------------------------------

    adjustment_dates = pd.to_datetime(
        adjustment["date_key"],
        errors="coerce"
    )

    candidates = adjustment[
        (adjustment_dates >=
         event_date - pd.Timedelta(days=60))
        &
        (adjustment_dates <=
         event_date + pd.Timedelta(days=60))
    ].copy()


    if len(candidates) > 0:

        candidates["distance_days"] = (
            pd.to_datetime(
                candidates["date_key"]
            ) - event_date
        ).abs().dt.days

        candidates = candidates.sort_values(
            ["distance_days"]
        )

        best = candidates.iloc[0]

        adjustment_date = best["date_key"]

        adjustment_factor = best[
            "adjustment_factor"
        ]

        adjustment_distance = int(
            best["distance_days"]
        )

        adjustment_status = best[
            adjustment_status_column
        ]

    else:

        adjustment_date = np.nan
        adjustment_factor = np.nan
        adjustment_distance = np.nan
        adjustment_status = (
            "NO_ADJUSTMENT_IN_60D"
        )


    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    abs_return = abs(event_raw_return)


    if (
        adjustment_status ==
        "EXPLAINED_BY_ADJUSTMENT"
        and not pd.isna(adjustment_distance)
        and adjustment_distance <= 30
    ):

        classification = (
            "ADJUSTMENT_SUPPORTED"
        )

    elif (
        event_no_trade_before >= 3
        or event_no_trade_after >= 3
    ):

        classification = (
            "REOPENING_WITHOUT_ADJUSTMENT"
        )

    elif abs_return >= 0.50:

        classification = (
            "POSSIBLE_STRUCTURAL_BREAK"
        )

    elif abs_return >= 0.20:

        classification = (
            "REVIEW_REQUIRED"
        )

    else:

        classification = (
            "NORMAL_EXTREME"
        )


    results.append(
        {
            "event_date":
                event["date_key"],

            "raw_close":
                event["raw_close"],

            "previous_close":
                event["previous_close"],

            "raw_return_1d":
                event_raw_return,

            "no_trade_before":
                event_no_trade_before,

            "no_trade_after":
                event_no_trade_after,

            "stage_20_11_classification":
                event["classification"],

            "adjustment_date":
                adjustment_date,

            "adjustment_factor":
                adjustment_factor,

            "adjustment_distance_days":
                adjustment_distance,

            "adjustment_status":
                adjustment_status,

            "final_classification":
                classification,
        }
    )


result_df = pd.DataFrame(results)


# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

print()
print(
    "=== STAGE 20.12: "
    "EXTREME ↔ CORPORATE ACTION RECONCILIATION ==="
)

print()

print(
    f"Extreme events analyzed: "
    f"{len(result_df)}"
)


print()
print("=== FINAL CLASSIFICATION ===")

print(
    result_df[
        "final_classification"
    ]
    .value_counts()
    .to_string()
)


print()
print("=== ADJUSTMENT MATCH STATUS ===")

print(
    result_df[
        "adjustment_status"
    ]
    .value_counts()
    .to_string()
)


print()
print("=== EVENTS WITH ADJUSTMENT MATCH ===")

matched = result_df[
    result_df["adjustment_date"].notna()
][
    [
        "event_date",
        "raw_return_1d",
        "adjustment_date",
        "adjustment_factor",
        "adjustment_distance_days",
        "adjustment_status",
        "final_classification",
    ]
]

if len(matched) > 0:

    print(
        matched.to_string(index=False)
    )

else:

    print(
        "No adjustment matches found."
    )


print()
print("=== POSSIBLE STRUCTURAL BREAKS ===")

structural = result_df[
    result_df["final_classification"]
    ==
    "POSSIBLE_STRUCTURAL_BREAK"
]

if len(structural) > 0:

    print(
        structural[
            [
                "event_date",
                "raw_return_1d",
                "no_trade_before",
                "no_trade_after",
                "adjustment_date",
                "adjustment_status",
            ]
        ].to_string(index=False)
    )

else:

    print("None")


print()
print("=== STAGE 20.12 SUMMARY ===")
print()

print(
    f"Total extreme events: "
    f"{len(result_df)}"
)

print(
    "Adjustment-supported: "
    f"{(result_df['final_classification'] == 'ADJUSTMENT_SUPPORTED').sum()}"
)

print(
    "Reopening without adjustment: "
    f"{(result_df['final_classification'] == 'REOPENING_WITHOUT_ADJUSTMENT').sum()}"
)

print(
    "Possible structural break: "
    f"{(result_df['final_classification'] == 'POSSIBLE_STRUCTURAL_BREAK').sum()}"
)

print(
    "Review required: "
    f"{(result_df['final_classification'] == 'REVIEW_REQUIRED').sum()}"
)

print(
    "Normal extreme: "
    f"{(result_df['final_classification'] == 'NORMAL_EXTREME').sum()}"
)


# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
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
    "This stage reconciles extreme events "
    "with corporate-action records."
)