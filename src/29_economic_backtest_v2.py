import pandas as pd
import numpy as np

from xgboost import XGBClassifier


# ============================================================
# QADSS - ECONOMIC EVALUATION / BACKTEST - V2 XGBOOST
# ============================================================

MODEL_FILE = (
    "data/processed/"
    "webmelat_model_ready_v2.csv"
)

FEATURE_FILE = (
    "data/processed/"
    "webmelat_features_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v2_economic_backtest.csv"
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

TARGET = "target"

PURGE = 5

HOLDING_PERIOD = 5


print("=" * 80)
print("QADSS - ECONOMIC EVALUATION / BACKTEST - V2 XGBOOST")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

model_df = pd.read_csv(
    MODEL_FILE
)

feature_df = pd.read_csv(
    FEATURE_FILE
)

print("\n")
print(f"Model-ready observations: {len(model_df)}")
print(f"Feature observations    : {len(feature_df)}")


# ============================================================
# 2. PREPARE DATES
# ============================================================

model_df["dEven"] = pd.to_datetime(
    model_df["dEven"],
    errors="coerce"
)

feature_df["dEven"] = pd.to_datetime(
    feature_df["dEven"],
    errors="coerce"
)


if model_df["dEven"].isna().any():

    raise ValueError(
        "Invalid dates found in model-ready data."
    )


if feature_df["dEven"].isna().any():

    raise ValueError(
        "Invalid dates found in feature data."
    )


# ============================================================
# 3. MERGE CLOSE PRICE
# ============================================================

price_df = feature_df[
    [
        "dEven",
        "close"
    ]
].copy()


price_df = price_df.sort_values(
    "dEven"
)


price_df = price_df.drop_duplicates(
    "dEven"
)


model_df = model_df.merge(
    price_df,
    on="dEven",
    how="left",
    validate="one_to_one"
)


if model_df["close"].isna().any():

    raise ValueError(
        "Some model dates do not have close prices."
    )


# ============================================================
# 4. CALCULATE 5-DAY FORWARD RETURN
# ============================================================

full_price = feature_df[
    [
        "dEven",
        "close"
    ]
].copy()


full_price = full_price.sort_values(
    "dEven"
).reset_index(
    drop=True
)


full_price["future_close_5d"] = (
    full_price["close"].shift(
        -HOLDING_PERIOD
    )
)


full_price["future_return_5d"] = (
    full_price["future_close_5d"]
    / full_price["close"]
    - 1
)


forward_return_map = full_price[
    [
        "dEven",
        "future_return_5d"
    ]
].copy()


model_df = model_df.merge(
    forward_return_map,
    on="dEven",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 5. REMOVE UNAVAILABLE FUTURE RETURNS
# ============================================================

model_df = model_df.dropna(
    subset=[
        "future_return_5d"
    ]
).copy()


# ============================================================
# 6. TIME-BASED SPLIT
# ============================================================

model_df = model_df.sort_values(
    "dEven"
).reset_index(
    drop=True
)


n = len(model_df)

split_index = int(
    n * 0.60
)

train_end = (
    split_index - PURGE
)


train_df = model_df.iloc[
    :train_end
].copy()


test_df = model_df.iloc[
    split_index:
].copy()


X_train = train_df[
    FEATURES
]

y_train = train_df[
    TARGET
].astype(int)


X_test = test_df[
    FEATURES
]

y_test = test_df[
    TARGET
].astype(int)


print("\n")
print("=" * 80)
print("TIME-BASED SPLIT")
print("=" * 80)

print(
    f"Training observations: {len(train_df)}"
)

print(
    f"Test observations    : {len(test_df)}"
)

print(
    f"Purged observations  : {PURGE}"
)


# ============================================================
# 7. TRAIN XGBOOST
# ============================================================

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


print("\n")
print("=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. PREDICT PROBABILITIES
# ============================================================

test_df["probability"] = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 9. BUY & HOLD BASELINE
# ============================================================

test_start_date = test_df[
    "dEven"
].min()

test_end_date = test_df[
    "dEven"
].max()


buy_hold_start = test_df[
    "close"
].iloc[0]

buy_hold_end = test_df[
    "close"
].iloc[-1]


buy_hold_return = (
    buy_hold_end
    / buy_hold_start
    - 1
)


# ============================================================
# 10. EVENT-BASED BACKTEST
# ============================================================

thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


results = []


for threshold in thresholds:

    signals = (
        test_df[
            "probability"
        ].values
        >= threshold
    )

    signal_indices = np.where(
        signals
    )[0]


    selected_returns = []

    selected_dates = []

    i = 0

    while i < len(test_df):

        if signals[i]:

            forward_return = (
                test_df[
                    "future_return_5d"
                ].iloc[i]
            )

            if pd.notna(
                forward_return
            ):

                selected_returns.append(
                    forward_return
                )

                selected_dates.append(
                    test_df[
                        "dEven"
                    ].iloc[i]
                )

            i += HOLDING_PERIOD

        else:

            i += 1


    selected_returns = np.array(
        selected_returns,
        dtype=float
    )


    if len(
        selected_returns
    ) > 0:

        cumulative_return = (
            np.prod(
                1 + selected_returns
            ) - 1
        )

        mean_return = (
            selected_returns.mean()
        )

        median_return = (
            np.median(
                selected_returns
            )
        )

        win_rate = (
            selected_returns > 0
        ).mean()

        worst_trade = (
            selected_returns.min()
        )

        best_trade = (
            selected_returns.max()
        )

    else:

        cumulative_return = np.nan

        mean_return = np.nan

        median_return = np.nan

        win_rate = np.nan

        worst_trade = np.nan

        best_trade = np.nan


    results.append(
        {
            "Threshold": threshold,

            "Test_Observations":
                len(test_df),

            "Trades":
                len(selected_returns),

            "Trade_Rate":
                len(selected_returns)
                / len(test_df),

            "Mean_5D_Return":
                mean_return,

            "Median_5D_Return":
                median_return,

            "Win_Rate":
                win_rate,

            "Cumulative_Return":
                cumulative_return,

            "Best_Trade":
                best_trade,

            "Worst_Trade":
                worst_trade,

            "Buy_Hold_Return":
                buy_hold_return
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("ECONOMIC BACKTEST RESULTS")
print("=" * 80)

print(
    f"\nTest period:"
)

print(
    f"Start: {test_start_date.date()}"
)

print(
    f"End  : {test_end_date.date()}"
)


print(
    f"\nBuy & Hold Return: "
    f"{buy_hold_return:.4f}"
)


print("\n")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. IMPORTANT INTERPRETATION
# ============================================================

print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "\nThis is a baseline economic evaluation."
)

print(
    "Transaction costs, slippage, taxes and "
    "position sizing are NOT included."
)

print(
    "\nThe 5-day holding period is aligned "
    "with the prediction target."
)

print(
    "Signals are non-overlapping to reduce "
    "double-counting of overlapping 5-day trades."
)

print(
    "\nThis result must NOT be interpreted as "
    "guaranteed future profitability."
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("=" * 80)