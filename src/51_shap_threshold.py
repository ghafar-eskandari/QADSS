import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score
)

import shap


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2.csv"
)

SHAP_OUTPUT = (
    BASE / "data" / "processed" /
    "stage23_shap_importance.csv"
)

THRESHOLD_OUTPUT = (
    BASE / "data" / "processed" /
    "stage23_threshold_analysis.csv"
)


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["dEven"],
    errors="coerce"
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


X = df[FEATURES]

y = df["target"].astype(int)


# =========================================================
# TEMPORAL SPLIT
# =========================================================

split_idx = int(
    len(df) * 0.60
)

PURGE = 5

train_end = split_idx - PURGE

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

test_dates = df["date"].iloc[split_idx:]


# =========================================================
# MODEL
# =========================================================

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


# =========================================================
# STAGE 23
# =========================================================

print()
print("=" * 70)
print("STAGE 23: SHAP + THRESHOLD ANALYSIS")
print("=" * 70)

print()
print(
    f"Train rows: {len(X_train)}"
)

print(
    f"Test rows: {len(X_test)}"
)


# =========================================================
# SHAP
# =========================================================

print()
print("=== SHAP IMPORTANCE ===")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_test
)

shap_values = np.asarray(
    shap_values
)

if shap_values.ndim == 3:

    shap_values = shap_values[:, :, 1]


mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)


shap_df = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
})


shap_df = (
    shap_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


shap_df["rank"] = (
    np.arange(len(shap_df)) + 1
)


print()

print(
    shap_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE SHAP
# =========================================================

shap_df.to_csv(
    SHAP_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

print()
print(
    "=== THRESHOLD ANALYSIS ==="
)

thresholds = np.arange(
    0.45,
    0.76,
    0.05
)

threshold_results = []


for threshold in thresholds:

    prediction = (
        probability >= threshold
    ).astype(int)

    signals = int(
        prediction.sum()
    )

    signal_rate = (
        signals /
        len(prediction)
    )

    if signals > 0:

        signal_accuracy = (
            y_test[prediction == 1]
            .mean()
        )

    else:

        signal_accuracy = np.nan


    accuracy = accuracy_score(
        y_test,
        prediction
    )

    balanced = (
        balanced_accuracy_score(
            y_test,
            prediction
        )
    )


    threshold_results.append({

        "threshold":
            round(float(threshold), 2),

        "signals":
            signals,

        "signal_rate":
            signal_rate,

        "signal_accuracy":
            signal_accuracy,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print()

print(
    threshold_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE THRESHOLD
# =========================================================

threshold_df.to_csv(
    THRESHOLD_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("STAGE 23 COMPLETED")
print("=" * 70)

print()

print(
    "Top 5 SHAP features:"
)

print()

print(
    shap_df.head(5).to_string(
        index=False
    )
)

print()

print(
    f"Saved SHAP: "
    f"{SHAP_OUTPUT}"
)

print(
    f"Saved Threshold: "
    f"{THRESHOLD_OUTPUT}"
)

print()
print(
    "NEXT: STAGE 24 - ECONOMIC + RISK EVALUATION"
)