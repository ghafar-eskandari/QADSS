# ============================================================
# QADSS - STAGE 20
# BACKTEST INTEGRITY CHECK - V2
# ============================================================

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "webmelat_model_ready_v2.csv"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "webmelat_features_v2.csv"
)

FEATURES = [
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
    "no_trade",
]

HOLDING_PERIOD = 5
PURGE = 5
THRESHOLD = 0.60


# ============================================================
# LOAD DATA
# ============================================================

model_df = pd.read_csv(MODEL_FILE)
feature_df = pd.read_csv(FEATURE_FILE)

model_df["dEven"] = pd.to_datetime(
    model_df["dEven"],
    errors="coerce"
)

feature_df["dEven"] = pd.to_datetime(
    feature_df["dEven"],
    errors="coerce"
)


# ============================================================
# RAW PRICE DATA CHECK
# ============================================================

print()
print("=" * 75)
print("QADSS - BACKTEST INTEGRITY CHECK - V2")
print("=" * 75)

print()
print("RAW FEATURE DATA CHECK")
print("-" * 75)

print(
    "Rows:",
    len(feature_df)
)

print(
    "Date range:",
    feature_df["dEven"].min().date(),
    "to",
    feature_df["dEven"].max().date()
)

print(
    "Missing close:",
    feature_df["close"].isna().sum()
)

print(
    "Zero close:",
    (feature_df["close"] == 0).sum()
)

print(
    "Negative close:",
    (feature_df["close"] < 0).sum()
)


# ============================================================
# PRICE ALIGNMENT
# ============================================================

price_df = feature_df[
    ["dEven", "close"]
].copy()

price_df = price_df.sort_values(
    "dEven"
).reset_index(drop=True)

price_df["future_close_5d"] = (
    price_df["close"].shift(-HOLDING_PERIOD)
)

price_df["future_return_5d"] = (
    price_df["future_close_5d"]
    / price_df["close"]
) - 1


# ============================================================
# EXTREME FUTURE RETURNS
# ============================================================

print()
print("=" * 75)
print("EXTREME 5-DAY RETURNS")
print("=" * 75)

extreme = price_df[
    price_df["future_return_5d"].abs() > 0.50
].copy()

print()

if len(extreme) == 0:

    print(
        "No 5-day return greater than +/-50% found."
    )

else:

    print(
        extreme[
            [
                "dEven",
                "close",
                "future_close_5d",
                "future_return_5d",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# EXTREME NEGATIVE RETURN CHECK
# ============================================================

print()
print("=" * 75)
print("EXTREME NEGATIVE RETURN CHECK")
print("=" * 75)

worst_events = price_df[
    price_df["future_return_5d"] <= -0.50
].copy()

print()

if len(worst_events) == 0:

    print(
        "No extreme negative 5-day return found."
    )

else:

    print(
        worst_events[
            [
                "dEven",
                "close",
                "future_close_5d",
                "future_return_5d",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# MERGE MODEL DATA WITH PRICE DATA
# ============================================================

df = model_df.merge(
    price_df[
        [
            "dEven",
            "close",
            "future_close_5d",
            "future_return_5d",
        ]
    ],
    on="dEven",
    how="left"
)

df = df.sort_values(
    "dEven"
).reset_index(drop=True)


# ============================================================
# CHECK MERGE
# ============================================================

print()
print("=" * 75)
print("MERGE CHECK")
print("=" * 75)

print()

print(
    "Merged rows:",
    len(df)
)

print(
    "Missing close after merge:",
    df["close"].isna().sum()
)

print(
    "Missing future return after merge:",
    df["future_return_5d"].isna().sum()
)


# ============================================================
# REMOVE INVALID MODEL PERIOD
# ============================================================

df = df.dropna(
    subset=[
        *FEATURES,
        "target",
        "close",
        "future_return_5d",
    ]
).reset_index(drop=True)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_index = int(
    len(df) * 0.60
)

train_end = split_index - PURGE

train_df = df.iloc[
    :train_end
].copy()

test_df = df.iloc[
    split_index:
].copy()


# ============================================================
# TRAIN XGBOOST
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

test_df["probability"] = (
    model.predict_proba(X_test)[:, 1]
)


# ============================================================
# RECONSTRUCT NON-OVERLAPPING SIGNALS
# ============================================================

signals = []

i = 0

while i < len(test_df):

    if (
        test_df["probability"].iloc[i]
        >= THRESHOLD
    ):

        signals.append(i)

        i += HOLDING_PERIOD

    else:

        i += 1


signals_df = test_df.iloc[
    signals
].copy()


# ============================================================
# SIGNAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("THRESHOLD 0.60 SIGNAL CHECK")
print("=" * 75)

print()

print(
    "Number of signals:",
    len(signals_df)
)

print(
    "Test period:",
    test_df["dEven"].iloc[0].date(),
    "to",
    test_df["dEven"].iloc[-1].date()
)


# ============================================================
# WORST SIGNAL RETURNS
# ============================================================

print()
print("=" * 75)
print("WORST SIGNAL RETURNS")
print("=" * 75)

worst_signals = signals_df.sort_values(
    "future_return_5d"
).head(10)

print()

print(
    worst_signals[
        [
            "dEven",
            "close",
            "future_close_5d",
            "future_return_5d",
            "probability",
            "target",
            "no_trade",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST SIGNAL RETURNS
# ============================================================

print()
print("=" * 75)
print("BEST SIGNAL RETURNS")
print("=" * 75)

best_signals = signals_df.sort_values(
    "future_return_5d",
    ascending=False
).head(10)

print()

print(
    best_signals[
        [
            "dEven",
            "close",
            "future_close_5d",
            "future_return_5d",
            "probability",
            "target",
            "no_trade",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# NO-TRADE SIGNAL CHECK
# ============================================================

print()
print("=" * 75)
print("NO-TRADE SIGNAL CHECK")
print("=" * 75)

no_trade_signals = signals_df[
    signals_df["no_trade"] == 1
]

print()

print(
    "Signals with no_trade=1:",
    len(no_trade_signals)
)

if len(no_trade_signals) > 0:

    print()

    print(
        no_trade_signals[
            [
                "dEven",
                "close",
                "future_close_5d",
                "future_return_5d",
                "probability",
            ]
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# TEST DATA PRICE CHECK
# ============================================================

print()
print("=" * 75)
print("TEST DATA PRICE CHECK")
print("=" * 75)

print()

print(
    "First test date:",
    test_df["dEven"].iloc[0].date()
)

print(
    "First test close:",
    test_df["close"].iloc[0]
)

print(
    "Last test date:",
    test_df["dEven"].iloc[-1].date()
)

print(
    "Last test close:",
    test_df["close"].iloc[-1]
)

buy_hold = (
    test_df["close"].iloc[-1]
    / test_df["close"].iloc[0]
) - 1

print(
    "Buy & Hold:",
    f"{buy_hold:.4f}"
)


# ============================================================
# DATA CONSISTENCY
# ============================================================

print()
print("=" * 75)
print("DATA CONSISTENCY")
print("=" * 75)

print()

print(
    "Model-ready rows:",
    len(model_df)
)

print(
    "Feature rows:",
    len(feature_df)
)

print(
    "Merged rows:",
    len(df)
)

print(
    "Train rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Signals:",
    len(signals_df)
)

print()

print(
    "Missing future returns:",
    df["future_return_5d"].isna().sum()
)

print(
    "Infinite future returns:",
    np.isinf(
        df["future_return_5d"]
    ).sum()
)

print()

print("=" * 75)
print("INTEGRITY CHECK COMPLETED")
print("=" * 75)