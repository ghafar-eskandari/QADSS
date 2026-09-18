import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE / "data" / "processed" /
    "webmelat_features_v2.csv"
)

MODEL_READY_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2.csv"
)

THRESHOLD_FILE = (
    BASE / "data" / "processed" /
    "stage23_threshold_analysis.csv"
)

OUTPUT_FILE = (
    BASE / "data" / "processed" /
    "stage24_economic_risk_evaluation.csv"
)


# =========================================================
# PARAMETERS
# =========================================================

THRESHOLDS = np.arange(
    0.50,
    0.76,
    0.05
)

INITIAL_CAPITAL = 1.0

# Simple transaction-cost scenarios
TRANSACTION_COSTS = [
    0.0000,
    0.0010,
    0.0020
]


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    MODEL_READY_FILE
)

features = pd.read_csv(
    INPUT_FILE
)


# =========================================================
# DATE
# =========================================================

df["date"] = pd.to_datetime(
    df["dEven"],
    errors="coerce"
)

features["date"] = pd.to_datetime(
    features["dEven"],
    errors="coerce"
)


# =========================================================
# SORT
# =========================================================

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

features = (
    features
    .sort_values("date")
    .reset_index(drop=True)
)


# =========================================================
# CHECK FUTURE RETURN
# =========================================================

if "future_return_5d" not in features.columns:

    features["future_return_5d"] = (
        features["close"]
        .shift(-5)
        /
        features["close"]
        - 1
    )


# =========================================================
# MERGE
# =========================================================

required_columns = [
    "date",
    "future_return_5d"
]

future_df = (
    features[required_columns]
    .drop_duplicates("date")
)


df = df.merge(
    future_df,
    on="date",
    how="left"
)


# =========================================================
# TEMPORAL SPLIT
# Same 60/40 structure as Stage 23
# =========================================================

split_idx = int(
    len(df) * 0.60
)

PURGE = 5

test = (
    df.iloc[split_idx:]
    .copy()
    .reset_index(drop=True)
)


# =========================================================
# LOAD THRESHOLD RESULTS
# =========================================================

threshold_table = pd.read_csv(
    THRESHOLD_FILE
)


# =========================================================
# STAGE 24 HEADER
# =========================================================

print()
print("=" * 70)
print("STAGE 24: ECONOMIC + RISK EVALUATION")
print("=" * 70)

print()
print(
    f"Test rows: {len(test)}"
)

print(
    f"Test period: "
    f"{test['date'].min().date()} "
    f"-> "
    f"{test['date'].max().date()}"
)


# =========================================================
# PROBABILITY RECONSTRUCTION
# =========================================================
#
# Stage 23 did not save probabilities.
# Therefore we retrain exactly the same XGBoost model
# using exactly the same train/test split.
# =========================================================

from xgboost import XGBClassifier


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
    "no_trade"
]


X = df[FEATURES]
y = df["target"].astype(int)


train_end = split_idx - PURGE

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]


model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

model.fit(
    X_train,
    y_train
)


probability = (
    model
    .predict_proba(X_test)[:, 1]
)


test["probability"] = probability


# =========================================================
# FUTURE RETURN CHECK
# =========================================================

test = test[
    test["future_return_5d"].notna()
].copy()


test = test[
    np.isfinite(
        test["future_return_5d"]
    )
].copy()


# =========================================================
# FUNCTION: MAX DRAWDOWN
# =========================================================

def max_drawdown(equity):

    equity = np.asarray(
        equity,
        dtype=float
    )

    if len(equity) == 0:
        return np.nan

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity /
        running_max
        - 1
    )

    return drawdown.min()


# =========================================================
# FUNCTION: SHARPE
# =========================================================

def sharpe_ratio(returns):

    returns = np.asarray(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return np.nan

    std = returns.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        returns.mean() /
        std
        *
        np.sqrt(len(returns))
    )


# =========================================================
# EVALUATION
# =========================================================

results = []


for threshold in THRESHOLDS:

    signal = (
        test["probability"]
        >= threshold
    ).astype(int)


    signal_returns = (
        test
        .loc[
            signal == 1,
            "future_return_5d"
        ]
        .copy()
    )


    signals = len(
        signal_returns
    )


    # -----------------------------------------------------
    # NO SIGNAL
    # -----------------------------------------------------

    if signals == 0:

        continue


    # -----------------------------------------------------
    # BASIC RETURN METRICS
    # -----------------------------------------------------

    mean_return = (
        signal_returns.mean()
    )

    median_return = (
        signal_returns.median()
    )

    win_rate = (
        signal_returns > 0
    ).mean()


    # -----------------------------------------------------
    # COMPOUNDED RETURN
    # -----------------------------------------------------

    cumulative_return = (
        (1 + signal_returns)
        .prod()
        - 1
    )


    # -----------------------------------------------------
    # PROFIT FACTOR
    # -----------------------------------------------------

    gross_profit = (
        signal_returns[
            signal_returns > 0
        ]
        .sum()
    )

    gross_loss = -(
        signal_returns[
            signal_returns < 0
        ]
        .sum()
    )


    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = np.nan


    # -----------------------------------------------------
    # SHARPE
    # -----------------------------------------------------

    sharpe = sharpe_ratio(
        signal_returns
    )


    # -----------------------------------------------------
    # EQUITY CURVE
    # -----------------------------------------------------

    equity = (
        1 +
        signal_returns
    ).cumprod()


    mdd = max_drawdown(
        equity
    )


    # -----------------------------------------------------
    # WORST TRADE
    # -----------------------------------------------------

    worst_trade = (
        signal_returns.min()
    )


    best_trade = (
        signal_returns.max()
    )


    # -----------------------------------------------------
    # BUY & HOLD
    # -----------------------------------------------------

    buy_hold_returns = (
        test["future_return_5d"]
        .dropna()
    )

    buy_hold_cumulative = (
        (1 + buy_hold_returns)
        .prod()
        - 1
    )


    # -----------------------------------------------------
    # TRANSACTION COST SCENARIOS
    # -----------------------------------------------------

    cost_results = {}

    for cost in TRANSACTION_COSTS:

        net_returns = (
            signal_returns
            - cost
        )

        net_cumulative = (
            (1 + net_returns)
            .prod()
            - 1
        )

        cost_results[
            f"net_return_cost_{cost:.4f}"
        ] = net_cumulative


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    row = {

        "threshold":
            round(
                float(threshold),
                2
            ),

        "signals":
            signals,

        "signal_rate":
            signals / len(test),

        "mean_5d_return":
            mean_return,

        "median_5d_return":
            median_return,

        "win_rate":
            win_rate,

        "cumulative_return":
            cumulative_return,

        "profit_factor":
            profit_factor,

        "sharpe":
            sharpe,

        "max_drawdown":
            mdd,

        "worst_trade":
            worst_trade,

        "best_trade":
            best_trade,

        "buy_hold_cumulative":
            buy_hold_cumulative
    }


    row.update(
        cost_results
    )


    results.append(
        row
    )


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print(
    "=== ECONOMIC + RISK RESULTS ==="
)

print()

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("STAGE 24 COMPLETED")
print("=" * 70)

print()

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Thresholds evaluated:"
)

print(
    ", ".join(
        [
            f"{x:.2f}"
            for x in THRESHOLDS
        ]
    )
)

print()

print(
    "NEXT: STAGE 25 - QADSS DECISION SUPPORT"
)

print("=" * 70)