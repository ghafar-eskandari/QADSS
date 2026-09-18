import pandas as pd
import json

MODEL = r".\data\processed\webmelat_model_ready_v2.csv"
EVENTS = r".\data\processed\corporate_action_residual_check.csv"
RAW = r".\data\raw\webmelat_tsetmc_raw.json"
OUTPUT = r".\data\processed\contaminated_target_impact.csv"

# Load model-ready data
df = pd.read_csv(
    MODEL,
    dtype={"dEven": str}
)

df["date"] = pd.to_datetime(
    df["dEven"],
    format="%Y-%m-%d"
)

df = df.sort_values("date").reset_index(drop=True)

# Reproduce Stage 19 split
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

# Load audited event dates
events = pd.read_csv(
    EVENTS,
    dtype={"event_date": str}
)

event_dates = pd.to_datetime(
    events["event_date"],
    format="%Y-%m-%d"
)

# Load raw data to reproduce the original 5-row target logic
with open(RAW, "r", encoding="utf-8") as f:
    raw = json.load(f)

price_df = pd.DataFrame(
    raw["closingPriceDaily"]
)

price_df["date"] = pd.to_datetime(
    price_df["dEven"].astype(str),
    format="%Y%m%d"
)

price_df = price_df.sort_values(
    "date"
).reset_index(drop=True)

price_df["close"] = price_df["pClosing"]

price_df["future_close_5d"] = (
    price_df["close"].shift(-5)
)

price_df["future_return_5d"] = (
    price_df["future_close_5d"]
    / price_df["close"]
    - 1
)

price_df["dEven"] = (
    price_df["date"]
    .dt.strftime("%Y-%m-%d")
)

# Merge target information
df = df.merge(
    price_df[
        [
            "dEven",
            "future_return_5d"
        ]
    ],
    on="dEven",
    how="left"
)

# Identify contaminated model observations
contaminated_dates = set()

for event_date in event_dates:

    event_positions = df.index[
        df["date"] == event_date
    ].tolist()

    if not event_positions:
        continue

    event_pos = event_positions[0]

    # Previous 5 observations have this event
    # somewhere inside their 5-row future target.
    for pos in range(
        max(0, event_pos - 5),
        event_pos
    ):
        contaminated_dates.add(
            df.loc[pos, "date"]
        )

df["target_contaminated"] = (
    df["date"].isin(contaminated_dates)
)

# Test data
test_df = df[
    df["split"] == "TEST"
].copy()

test_contaminated = test_df[
    test_df["target_contaminated"]
].copy()

test_clean = test_df[
    ~test_df["target_contaminated"]
].copy()

# Summary
print()
print("=== STAGE 20.5: CONTAMINATED TARGET IMPACT ===")
print()

print("TOTAL DATA")
print(
    "Model-ready observations:",
    len(df)
)

print(
    "Contaminated observations:",
    int(
        df["target_contaminated"].sum()
    )
)

print()
print("=== BY SPLIT ===")

print(
    df.groupby(
        ["split", "target_contaminated"]
    )["target"]
    .agg(["count", "sum"])
    .to_string()
)

print()
print("=== TEST IMPACT ===")

print(
    "TEST observations:",
    len(test_df)
)

print(
    "Contaminated TEST:",
    len(test_contaminated)
)

print(
    "Clean TEST:",
    len(test_clean)
)

print(
    "Contamination rate:",
    round(
        len(test_contaminated)
        / len(test_df),
        4
    )
)

print()
print("=== TARGET DISTRIBUTION ===")

print()
print("Contaminated TEST:")

print(
    test_contaminated[
        "target"
    ].value_counts()
    .sort_index()
)

print()
print("Clean TEST:")

print(
    test_clean[
        "target"
    ].value_counts()
    .sort_index()
)

print()
print("=== CONTAMINATED TEST OBSERVATIONS ===")

print(
    test_contaminated[
        [
            "dEven",
            "target",
            "future_return_5d"
        ]
    ].to_string(index=False)
)

# Save
df[
    [
        "dEven",
        "split",
        "target",
        "future_return_5d",
        "target_contaminated"
    ]
].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Saved:", OUTPUT)
