import pandas as pd
import json

MODEL = r".\data\processed\webmelat_model_ready_v2.csv"
EVENTS = r".\data\processed\corporate_action_residual_check.csv"
RAW = r".\data\raw\webmelat_tsetmc_raw.json"
OUTPUT = r".\data\processed\target_contamination_audit.csv"

# Load model-ready data
model_df = pd.read_csv(
    MODEL,
    dtype={"dEven": str}
)

# Load events
events = pd.read_csv(
    EVENTS,
    dtype={"event_date": str}
)

# Load raw TSETMC prices
with open(RAW, "r", encoding="utf-8") as f:
    raw = json.load(f)

price_df = pd.DataFrame(raw["closingPriceDaily"])

price_df["date"] = pd.to_datetime(
    price_df["dEven"].astype(str),
    format="%Y%m%d"
)

price_df = price_df.sort_values("date").reset_index(drop=True)

# Recalculate 5-row future return from raw closing price
price_df["close"] = price_df["pClosing"]

price_df["future_close_5d"] = (
    price_df["close"].shift(-5)
)

price_df["future_return_5d"] = (
    price_df["future_close_5d"]
    / price_df["close"]
    - 1
)

# Convert date to same format as model-ready
price_df["dEven"] = price_df["date"].dt.strftime("%Y-%m-%d")

# Merge model rows with recalculated future returns
df = model_df.merge(
    price_df[
        [
            "dEven",
            "close",
            "future_close_5d",
            "future_return_5d"
        ]
    ],
    on="dEven",
    how="left"
)

df["date"] = pd.to_datetime(
    df["dEven"],
    format="%Y-%m-%d"
)

df = df.sort_values("date").reset_index(drop=True)

# Stage 19 split
TRAIN_SIZE = 2387
PURGE_SIZE = 5

df["split"] = "TEST"

df.loc[
    :TRAIN_SIZE - 1,
    "split"
] = "TRAIN"

df.loc[
    TRAIN_SIZE:TRAIN_SIZE + PURGE_SIZE - 1,
    "split"
] = "PURGE"

events["event_dt"] = pd.to_datetime(
    events["event_date"],
    format="%Y-%m-%d"
)

rows = []

for _, e in events.iterrows():

    event_date = e["event_dt"]

    event_rows = df.index[
        df["date"] == event_date
    ].tolist()

    if not event_rows:

        rows.append({
            "event_date": e["event_date"],
            "residual_status": e["residual_status"],
            "raw_return_event": e["raw_return"],
            "event_row_exists": False,
            "target_affected_rows": 0,
            "train_affected": 0,
            "purge_affected": 0,
            "test_affected": 0
        })

        continue

    event_pos = event_rows[0]

    # The event enters the 5-row future target
    # of the previous five observations.
    affected_positions = range(
        max(0, event_pos - 5),
        event_pos
    )

    affected = df.iloc[
        list(affected_positions)
    ].copy()

    rows.append({

        "event_date":
            e["event_date"],

        "residual_status":
            e["residual_status"],

        "raw_return_event":
            e["raw_return"],

        "event_row_exists":
            True,

        "event_row_position":
            event_pos,

        "event_split":
            df.loc[event_pos, "split"],

        "event_target":
            df.loc[event_pos, "target"],

        "target_affected_rows":
            len(affected),

        "first_affected_date":
            affected["dEven"].iloc[0]
            if len(affected) > 0 else None,

        "last_affected_date":
            affected["dEven"].iloc[-1]
            if len(affected) > 0 else None,

        "train_affected":
            int(
                (affected["split"] == "TRAIN").sum()
            ),

        "purge_affected":
            int(
                (affected["split"] == "PURGE").sum()
            ),

        "test_affected":
            int(
                (affected["split"] == "TEST").sum()
            ),

        "max_abs_future_return_affected":
            affected[
                "future_return_5d"
            ].abs().max()
            if len(affected) > 0 else None
    })


audit = pd.DataFrame(rows)

audit.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== STAGE 20.4: TARGET CONTAMINATION AUDIT ===")
print()

print(audit.to_string(index=False))

print()
print("=== SUMMARY ===")

print(
    "Total events:",
    len(audit)
)

print(
    "Events with existing row:",
    int(
        audit["event_row_exists"].sum()
    )
)

print(
    "Total target-affected rows:",
    int(
        audit["target_affected_rows"].sum()
    )
)

print(
    "Train affected:",
    int(
        audit["train_affected"].sum()
    )
)

print(
    "Purge affected:",
    int(
        audit["purge_affected"].sum()
    )
)

print(
    "Test affected:",
    int(
        audit["test_affected"].sum()
    )
)

print()
print("=== EVENTS AFFECTING TEST TARGETS ===")

print(
    audit[
        audit["test_affected"] > 0
    ].to_string(index=False)
)

print()
print("Saved:", OUTPUT)
