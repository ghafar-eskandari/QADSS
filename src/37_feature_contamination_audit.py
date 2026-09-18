import pandas as pd
import json

FEATURES = r".\data\processed\webmelat_features_v2.csv"
EVENTS = r".\data\processed\corporate_action_residual_check.csv"
RAW = r".\data\raw\webmelat_tsetmc_raw.json"
OUTPUT = r".\data\processed\feature_contamination_audit.csv"

# Load V2 features
df = pd.read_csv(
    FEATURES,
    dtype={"dEven": str}
)

df["date"] = pd.to_datetime(
    df["dEven"],
    format="%Y-%m-%d"
)

df = df.sort_values("date").reset_index(drop=True)

# Load audited events
events = pd.read_csv(
    EVENTS,
    dtype={"event_date": str}
)

events["event_dt"] = pd.to_datetime(
    events["event_date"],
    format="%Y-%m-%d"
)

# Feature columns
FEATURE_COLS = [
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

# Load raw data for exact event positions
with open(RAW, "r", encoding="utf-8") as f:
    raw = json.load(f)

raw_df = pd.DataFrame(
    raw["closingPriceDaily"]
)

raw_df["date"] = pd.to_datetime(
    raw_df["dEven"].astype(str),
    format="%Y%m%d"
)

raw_df = raw_df.sort_values(
    "date"
).reset_index(drop=True)

# --------------------------------------------------
# Audit:
# For each event, inspect the 20 observations AFTER
# and BEFORE the event.
# --------------------------------------------------

rows = []

for _, e in events.iterrows():

    event_date = e["event_dt"]

    event_positions = df.index[
        df["date"] == event_date
    ].tolist()

    if not event_positions:
        continue

    pos = event_positions[0]

    # 20 observations before + event + 20 after
    start = max(0, pos - 20)
    end = min(len(df), pos + 21)

    window = df.iloc[
        start:end
    ].copy()

    for _, r in window.iterrows():

        distance = (
            df.index[
                df["date"] == r["date"]
            ][0] - pos
        )

        record = {
            "event_date":
                e["event_date"],

            "residual_status":
                e["residual_status"],

            "observation_date":
                r["dEven"],

            "row_distance":
                distance
        }

        for feature in FEATURE_COLS:
            record[feature] = r[feature]

        rows.append(record)

audit = pd.DataFrame(rows)

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

print()
print("=== STAGE 20.7: FEATURE CONTAMINATION AUDIT ===")
print()

print(
    "Events analysed:",
    events.shape[0]
)

print(
    "Feature-window observations:",
    len(audit)
)

print()
print("=== EXTREME FEATURE VALUES ===")

for feature in FEATURE_COLS:

    s = audit[feature].dropna()

    if len(s) == 0:
        continue

    print(
        f"{feature:20s} "
        f"min={s.min(): .4f} "
        f"max={s.max(): .4f}"
    )

print()
print("=== EVENT-DAY FEATURE VALUES ===")

event_day = audit[
    audit["row_distance"] == 0
]

print(
    event_day[
        ["event_date"]
        + FEATURE_COLS
    ].to_string(index=False)
)

print()
print("=== EXTREME RETURNS AROUND EVENTS ===")

return_cols = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d"
]

for feature in return_cols:

    extreme = audit[
        audit[feature].abs() >= 0.20
    ]

    print()
    print(
        feature,
        "extreme rows:",
        len(extreme)
    )

    if len(extreme) > 0:
        print(
            extreme[
                [
                    "event_date",
                    "observation_date",
                    "row_distance",
                    feature
                ]
            ].to_string(index=False)
        )

# Save row-level audit
audit.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Saved:", OUTPUT)
