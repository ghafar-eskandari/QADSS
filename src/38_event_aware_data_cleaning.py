import pandas as pd
import json

FEATURES = r".\data\processed\webmelat_features_v2.csv"
EVENTS = r".\data\processed\corporate_action_residual_check.csv"
RAW = r".\data\raw\webmelat_tsetmc_raw.json"

OUTPUT = r".\data\processed\event_aware_data_flags.csv"


# ============================================================
# STAGE 20.8
# EVENT-AWARE DATA CLEANING
# ============================================================

print()
print("=== STAGE 20.8: EVENT-AWARE DATA CLEANING ===")
print()


# ------------------------------------------------------------
# 1. Load feature data
# ------------------------------------------------------------

df = pd.read_csv(
    FEATURES,
    dtype={"dEven": str}
)

df["date"] = pd.to_datetime(
    df["dEven"],
    format="%Y-%m-%d"
)

df = (
    df.sort_values("date")
      .reset_index(drop=True)
)


# ------------------------------------------------------------
# 2. Load event audit
# ------------------------------------------------------------

events = pd.read_csv(
    EVENTS,
    dtype={"event_date": str}
)

events["event_dt"] = pd.to_datetime(
    events["event_date"],
    format="%Y-%m-%d"
)


# ------------------------------------------------------------
# 3. Load raw TSETMC data
# ------------------------------------------------------------

with open(RAW, "r", encoding="utf-8") as f:
    raw = json.load(f)

raw_df = pd.DataFrame(
    raw["closingPriceDaily"]
)

raw_df["date"] = pd.to_datetime(
    raw_df["dEven"].astype(str),
    format="%Y%m%d"
)

raw_df = (
    raw_df.sort_values("date")
          .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Define event classification
# ------------------------------------------------------------

def classify_event(row):

    status = row["residual_status"]

    if status == "EXPLAINED_BY_ADJUSTMENT":
        return "ADJUSTMENT_EXPLAINED"

    if status == "UNEXPLAINED_AFTER_ADJUSTMENT":
        return "UNRESOLVED_EVENT"

    if status == "NO_OFFICIAL_ADJUSTMENT":
        return "UNRESOLVED_EVENT"

    return "REVIEW"


# ------------------------------------------------------------
# 5. Build event windows
# ------------------------------------------------------------
#
# We do NOT delete observations.
#
# We flag:
#
#   - 20 observations before event
#   - event day
#   - 20 observations after event
#
# This allows later sensitivity analysis.
# ------------------------------------------------------------

rows = []

for _, event in events.iterrows():

    event_date = event["event_dt"]

    positions = df.index[
        df["date"] == event_date
    ].tolist()

    if not positions:
        continue

    event_pos = positions[0]

    event_class = classify_event(event)

    start = max(
        0,
        event_pos - 20
    )

    end = min(
        len(df),
        event_pos + 21
    )

    for pos in range(start, end):

        r = df.iloc[pos]

        distance = pos - event_pos

        if distance < 0:
            window_type = "PRE_EVENT"

        elif distance == 0:
            window_type = "EVENT_DAY"

        else:
            window_type = "POST_EVENT"

        # Conservative flag:
        # all observations inside the event window
        # are marked as event-affected.
        #
        # We are NOT deleting them here.

        rows.append(
            {
                "event_date": event["event_date"],
                "event_class": event_class,
                "residual_status": event["residual_status"],
                "observation_date": r["dEven"],
                "row_distance": distance,
                "window_type": window_type,
                "event_affected": 1
            }
        )


# ------------------------------------------------------------
# 6. Create flag dataframe
# ------------------------------------------------------------

event_flags = pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. Merge flags onto complete feature dataset
# ------------------------------------------------------------

flag_columns = [
    "observation_date",
    "event_date",
    "event_class",
    "residual_status",
    "row_distance",
    "window_type",
    "event_affected"
]

event_flags = event_flags[
    flag_columns
]


# A date can theoretically belong to more than one event window.
# Keep the closest event.

event_flags["abs_distance"] = (
    event_flags["row_distance"].abs()
)

event_flags = (
    event_flags
    .sort_values(
        [
            "observation_date",
            "abs_distance"
        ]
    )
    .drop_duplicates(
        subset=["observation_date"],
        keep="first"
    )
)


clean_df = df[
    ["dEven"]
].copy()

clean_df["event_affected"] = 0
clean_df["event_class"] = "CLEAN"
clean_df["event_date"] = pd.NA
clean_df["residual_status"] = pd.NA
clean_df["row_distance"] = pd.NA
clean_df["window_type"] = "NORMAL"


# ------------------------------------------------------------
# 8. Apply event flags
# ------------------------------------------------------------

for _, flag in event_flags.iterrows():

    mask = (
        clean_df["dEven"]
        == flag["observation_date"]
    )

    clean_df.loc[
        mask,
        "event_affected"
    ] = 1

    clean_df.loc[
        mask,
        "event_class"
    ] = flag["event_class"]

    clean_df.loc[
        mask,
        "event_date"
    ] = flag["event_date"]

    clean_df.loc[
        mask,
        "residual_status"
    ] = flag["residual_status"]

    clean_df.loc[
        mask,
        "row_distance"
    ] = flag["row_distance"]

    clean_df.loc[
        mask,
        "window_type"
    ] = flag["window_type"]


# ------------------------------------------------------------
# 9. Summary
# ------------------------------------------------------------

print("Total feature rows:", len(clean_df))

print(
    "Event-affected rows:",
    int(clean_df["event_affected"].sum())
)

print(
    "Clean rows:",
    int(
        (clean_df["event_affected"] == 0).sum()
    )
)

print()

print("=== EVENT CLASS SUMMARY ===")

print(
    clean_df[
        clean_df["event_affected"] == 1
    ]["event_class"]
    .value_counts()
    .to_string()
)

print()

print("=== WINDOW TYPE SUMMARY ===")

print(
    clean_df[
        clean_df["event_affected"] == 1
    ]["window_type"]
    .value_counts()
    .to_string()
)

print()

print("=== EVENT SUMMARY ===")

event_summary = (
    clean_df[
        clean_df["event_affected"] == 1
    ]
    .groupby(
        [
            "event_date",
            "event_class"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="affected_rows"
    )
)

print(
    event_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------

clean_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Saved:", OUTPUT)
print()
print("IMPORTANT:")
print("No rows were deleted.")
print("Only event-affected observations were flagged.")
print()