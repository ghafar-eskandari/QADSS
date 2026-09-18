import pandas as pd
import numpy as np
import json

CLEAN_DATA = r".\data\processed\webmelat_model_ready_v2_clean.csv"
AUDIT_FILE = r".\data\processed\clean_data_audit.csv"
RAW_FILE = r".\data\raw\webmelat_tsetmc_raw.json"
ADJUSTMENT_FILE = r".\data\processed\corporate_action_residual_check.csv"

OUTPUT = r".\data\processed\extreme_event_classification.csv"

print()
print("=== STAGE 20.11: EXTREME EVENT CLASSIFICATION ===")
print()

# ---------------------------------------------------------
# 1. Load clean dataset
# ---------------------------------------------------------

df = pd.read_csv(
    CLEAN_DATA,
    dtype={"dEven": str}
)

df["dEven"] = (
    pd.to_datetime(
        df["dEven"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

print("Clean dataset rows:", len(df))

# ---------------------------------------------------------
# 2. Load extreme audit
# ---------------------------------------------------------

audit = pd.read_csv(
    AUDIT_FILE,
    dtype={"dEven": str}
)

audit["dEven"] = (
    pd.to_datetime(
        audit["dEven"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

extreme_dates = (
    audit["dEven"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

print("Extreme dates:", len(extreme_dates))

# ---------------------------------------------------------
# 3. Load raw TSETMC data
# ---------------------------------------------------------

with open(
    RAW_FILE,
    "r",
    encoding="utf-8"
) as f:
    raw = json.load(f)

raw_df = pd.DataFrame(
    raw["closingPriceDaily"]
)

raw_df["dEven"] = (
    raw_df["dEven"]
    .astype(str)
    .str.strip()
)

# Convert raw TSETMC date to same YYYY-MM-DD format
raw_df["date"] = pd.to_datetime(
    raw_df["dEven"],
    format="%Y%m%d",
    errors="coerce"
)

raw_df["date_key"] = (
    raw_df["date"]
    .dt.strftime("%Y-%m-%d")
)

raw_df = raw_df.sort_values(
    "date"
).reset_index(drop=True)

numeric_cols = [
    "pClosing",
    "priceYesterday",
    "priceFirst",
    "priceMin",
    "priceMax",
    "zTotTran",
    "qTotTran5J",
    "qTotCap"
]

for col in numeric_cols:
    raw_df[col] = pd.to_numeric(
        raw_df[col],
        errors="coerce"
    )

raw_df["no_trade"] = (
    raw_df["zTotTran"] == 0
).astype(int)

print("Raw rows:", len(raw_df))

# ---------------------------------------------------------
# 4. Load corporate-action audit
# ---------------------------------------------------------

adjustments = pd.read_csv(
    ADJUSTMENT_FILE
)

adjustments["event_date"] = (
    pd.to_datetime(
        adjustments["event_date"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

adjustment_map = {}

for _, row in adjustments.iterrows():

    adjustment_map[
        row["event_date"]
    ] = row.get(
        "residual_status",
        ""
    )

# ---------------------------------------------------------
# 5. Build classification
# ---------------------------------------------------------

results = []

for extreme_date in extreme_dates:

    match = raw_df[
        raw_df["date_key"] == extreme_date
    ]

    if len(match) == 0:
        continue

    idx = match.index[0]

    row = raw_df.loc[idx]

    # -----------------------------------------------------
    # Previous and next raw observations
    # -----------------------------------------------------

    previous_close = np.nan
    next_close = np.nan

    if idx > 0:
        previous_close = raw_df.loc[
            idx - 1,
            "pClosing"
        ]

    if idx < len(raw_df) - 1:
        next_close = raw_df.loc[
            idx + 1,
            "pClosing"
        ]

    # -----------------------------------------------------
    # Local 10-observation window
    # -----------------------------------------------------

    start_idx = max(
        0,
        idx - 10
    )

    end_idx = min(
        len(raw_df),
        idx + 11
    )

    local_window = raw_df.iloc[
        start_idx:end_idx
    ]

    no_trade_count = int(
        local_window["no_trade"].sum()
    )

    traded_count = int(
        (local_window["no_trade"] == 0).sum()
    )

    # -----------------------------------------------------
    # Consecutive no-trade before
    # -----------------------------------------------------

    no_trade_before = 0

    j = idx - 1

    while j >= 0:

        if raw_df.loc[
            j,
            "no_trade"
        ] == 1:

            no_trade_before += 1
            j -= 1

        else:
            break

    # -----------------------------------------------------
    # Consecutive no-trade after
    # -----------------------------------------------------

    no_trade_after = 0

    j = idx + 1

    while j < len(raw_df):

        if raw_df.loc[
            j,
            "no_trade"
        ] == 1:

            no_trade_after += 1
            j += 1

        else:
            break

    # -----------------------------------------------------
    # Raw one-day return
    # -----------------------------------------------------

    if (
        pd.notna(previous_close)
        and previous_close != 0
    ):

        raw_return_1d = (
            row["pClosing"] /
            previous_close
        ) - 1

    else:

        raw_return_1d = np.nan

    # -----------------------------------------------------
    # Adjustment status
    # -----------------------------------------------------

    adjustment_status = adjustment_map.get(
        extreme_date,
        "NO_OFFICIAL_ADJUSTMENT"
    )

    # -----------------------------------------------------
    # Preliminary classification
    # -----------------------------------------------------

    if (
        abs(raw_return_1d) >= 0.50
        and (
            no_trade_before >= 3
            or no_trade_after >= 3
        )
    ):

        classification = (
            "POSSIBLE_STRUCTURAL_BREAK"
        )

    elif (
        abs(raw_return_1d) >= 0.50
        and adjustment_status
        != "NO_OFFICIAL_ADJUSTMENT"
    ):

        classification = (
            "ADJUSTMENT_RELATED"
        )

    elif (
        no_trade_before >= 3
        or no_trade_after >= 3
    ):

        classification = (
            "REOPENING_RELATED"
        )

    elif abs(raw_return_1d) >= 0.50:

        classification = (
            "REVIEW_REQUIRED"
        )

    else:

        classification = (
            "NORMAL_EXTREME"
        )

    results.append(
        {
            "dEven": extreme_date,
            "raw_close": row["pClosing"],
            "previous_close": previous_close,
            "next_close": next_close,
            "raw_return_1d": raw_return_1d,
            "price_min": row["priceMin"],
            "price_max": row["priceMax"],
            "zTotTran": row["zTotTran"],
            "qTotTran5J": row["qTotTran5J"],
            "no_trade": row["no_trade"],
            "no_trade_before": no_trade_before,
            "no_trade_after": no_trade_after,
            "local_no_trade_count": no_trade_count,
            "local_traded_count": traded_count,
            "adjustment_status": adjustment_status,
            "classification": classification
        }
    )

# ---------------------------------------------------------
# 6. Create result dataframe
# ---------------------------------------------------------

result_df = pd.DataFrame(results)

if len(result_df) > 0:

    result_df = result_df.sort_values(
        "dEven"
    ).reset_index(drop=True)

# ---------------------------------------------------------
# 7. Print detailed results
# ---------------------------------------------------------

print()
print("=== EXTREME EVENT CLASSIFICATION ===")

if len(result_df) > 0:

    print(
        result_df[
            [
                "dEven",
                "raw_close",
                "previous_close",
                "raw_return_1d",
                "no_trade_before",
                "no_trade_after",
                "adjustment_status",
                "classification"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "WARNING: No extreme dates matched raw TSETMC data."
    )

# ---------------------------------------------------------
# 8. Classification summary
# ---------------------------------------------------------

print()
print("=== CLASSIFICATION SUMMARY ===")

if len(result_df) > 0:

    print(
        result_df[
            "classification"
        ]
        .value_counts()
        .to_string()
    )

# ---------------------------------------------------------
# 9. Adjustment summary
# ---------------------------------------------------------

print()
print("=== ADJUSTMENT STATUS SUMMARY ===")

if len(result_df) > 0:

    print(
        result_df[
            "adjustment_status"
        ]
        .value_counts()
        .to_string()
    )

# ---------------------------------------------------------
# 10. Save
# ---------------------------------------------------------

result_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== STAGE 20.11 SUMMARY ===")
print()

print(
    "Extreme dates classified:",
    len(result_df)
)

print(
    "Saved classification:",
    OUTPUT
)

print()
print("IMPORTANT:")
print("No rows were deleted.")
print("No model was trained.")
print(
    "Classification is an audit aid, "
    "not a final economic judgment."
)