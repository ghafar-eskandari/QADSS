# ============================================================
# QADSS - STAGE 19
# RISK & PERFORMANCE ANALYSIS - V2 XGBOOST
# ============================================================

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "v2_risk_performance.csv"
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

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

HOLDING_PERIOD = 5
PURGE = 5


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

feature_df = feature_df[
    ["dEven", "close"]
].copy()

df = model_df.merge(
    feature_df,
    on="dEven",
    how="left"
)

df = df.sort_values("dEven").reset_index(drop=True)


# ============================================================
# FUTURE RETURN
# ============================================================

df["future_return_5d"] = (
    df["close"].shift(-HOLDING_PERIOD) / df["close"] - 1
)

df = df.dropna(
    subset=FEATURES + ["target", "close", "future_return_5d"]
).reset_index(drop=True)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_index = int(len(df) * 0.60)

train_end = split_index - PURGE

train_df = df.iloc[:train_end].copy()
test_df = df.iloc[split_index:].copy()


X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]


# ============================================================
# TRAIN XGBOOST
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

model.fit(X_train, y_train)

test_df["probability"] = model.predict_proba(X_test)[:, 1]


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(returns):

    returns = np.asarray(returns, dtype=float)

    if len(returns) == 0:
        return {
            "Trades": 0,
            "Win_Rate": np.nan,
            "Mean_Return": np.nan,
            "Median_Return": np.nan,
            "Profit_Factor": np.nan,
            "Sharpe": np.nan,
            "Max_Drawdown": np.nan,
            "Cumulative_Return": np.nan,
            "Best_Trade": np.nan,
            "Worst_Trade": np.nan,
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = np.mean(returns > 0)

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    mean_return = returns.mean()

    std_return = returns.std(ddof=1)

    if std_return > 0:
        sharpe = (
            mean_return / std_return
        ) * np.sqrt(len(returns))
    else:
        sharpe = np.nan

    equity = np.cumprod(1 + returns)

    running_max = np.maximum.accumulate(equity)

    drawdown = (
        equity / running_max
    ) - 1

    max_drawdown = drawdown.min()

    cumulative_return = equity[-1] - 1

    return {
        "Trades": len(returns),
        "Win_Rate": win_rate,
        "Mean_Return": mean_return,
        "Median_Return": np.median(returns),
        "Profit_Factor": profit_factor,
        "Sharpe": sharpe,
        "Max_Drawdown": max_drawdown,
        "Cumulative_Return": cumulative_return,
        "Best_Trade": returns.max(),
        "Worst_Trade": returns.min(),
    }


# ============================================================
# RUN ANALYSIS
# ============================================================

results = []


for threshold in THRESHOLDS:

    probabilities = test_df["probability"].values
    future_returns = test_df["future_return_5d"].values

    selected_returns = []

    i = 0

    while i < len(test_df):

        if probabilities[i] >= threshold:

            selected_returns.append(
                future_returns[i]
            )

            i += HOLDING_PERIOD

        else:

            i += 1


    metrics = calculate_metrics(
        selected_returns
    )

    metrics["Threshold"] = threshold
    metrics["Test_Observations"] = len(test_df)

    if len(test_df) > 0:
        metrics["Trade_Rate"] = (
            metrics["Trades"] / len(test_df)
        )
    else:
        metrics["Trade_Rate"] = np.nan

    results.append(metrics)


# ============================================================
# BUY & HOLD
# ============================================================

buy_hold_return = (
    test_df["close"].iloc[-1]
    / test_df["close"].iloc[0]
) - 1


results_df = pd.DataFrame(results)

results_df["Buy_Hold_Return"] = buy_hold_return


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("QADSS - RISK & PERFORMANCE ANALYSIS - V2 XGBOOST")
print("=" * 70)

print()

print("Test period:")
print(
    "Start:",
    test_df["dEven"].iloc[0].date()
)

print(
    "End  :",
    test_df["dEven"].iloc[-1].date()
)

print()

print(
    "Buy & Hold Return:",
    f"{buy_hold_return:.4f}"
)

print()

print("=" * 70)
print("RISK & PERFORMANCE RESULTS")
print("=" * 70)

print()

display_columns = [
    "Threshold",
    "Trades",
    "Trade_Rate",
    "Win_Rate",
    "Mean_Return",
    "Median_Return",
    "Profit_Factor",
    "Sharpe",
    "Max_Drawdown",
    "Cumulative_Return",
    "Best_Trade",
    "Worst_Trade",
    "Buy_Hold_Return",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print()
print(
    "Saved to:"
)
print(
    "data/processed/v2_risk_performance.csv"
)

print()