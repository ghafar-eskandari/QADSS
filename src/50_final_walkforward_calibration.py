import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2.csv"
)

OUTPUT_FILE = (
    BASE / "data" / "processed" /
    "stage22_walkforward_results.csv"
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
# PARAMETERS
# =========================================================

N_FOLDS = 5

PURGE = 5

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1
}


# =========================================================
# WALK-FORWARD
# =========================================================

print()
print("=" * 70)
print("STAGE 22: FINAL WALK-FORWARD + CALIBRATION")
print("=" * 70)

print()
print(f"Rows: {len(df)}")
print(f"Features: {len(FEATURES)}")
print(f"Folds: {N_FOLDS}")
print(f"Purge: {PURGE}")


fold_size = len(df) // (N_FOLDS + 1)

results = []

all_test_probability = []
all_test_actual = []


for fold in range(N_FOLDS):

    train_end = (
        fold_size * (fold + 1)
    )

    test_start = train_end + PURGE

    test_end = (
        test_start + fold_size
    )

    if test_end > len(df):

        test_end = len(df)

    train_idx = np.arange(
        0,
        train_end
    )

    test_idx = np.arange(
        test_start,
        test_end
    )

    if len(test_idx) < 10:

        continue

    X_train = X.iloc[train_idx]

    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]

    y_test = y.iloc[test_idx]


    model = XGBClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )


    probability = (
        model
        .predict_proba(X_test)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)


    auc = roc_auc_score(
        y_test,
        probability
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    balanced = balanced_accuracy_score(
        y_test,
        prediction
    )


    results.append({
        "fold": fold + 1,

        "train_start":
            df.iloc[train_idx[0]]["date"].strftime(
                "%Y-%m-%d"
            ),

        "train_end":
            df.iloc[train_idx[-1]]["date"].strftime(
                "%Y-%m-%d"
            ),

        "test_start":
            df.iloc[test_idx[0]]["date"].strftime(
                "%Y-%m-%d"
            ),

        "test_end":
            df.iloc[test_idx[-1]]["date"].strftime(
                "%Y-%m-%d"
            ),

        "train_rows":
            len(train_idx),

        "test_rows":
            len(test_idx),

        "roc_auc":
            auc,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced
    })


    all_test_probability.extend(
        probability
    )

    all_test_actual.extend(
        y_test
    )


# =========================================================
# WALK-FORWARD RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)


print()
print(
    "=== WALK-FORWARD RESULTS ==="
)

print()

print(
    results_df.to_string(
        index=False
    )
)


mean_auc = (
    results_df["roc_auc"]
    .mean()
)

mean_accuracy = (
    results_df["accuracy"]
    .mean()
)

mean_balanced = (
    results_df["balanced_accuracy"]
    .mean()
)


print()
print(
    "=== WALK-FORWARD MEANS ==="
)

print(
    f"Mean ROC-AUC: "
    f"{mean_auc:.6f}"
)

print(
    f"Mean Accuracy: "
    f"{mean_accuracy:.6f}"
)

print(
    f"Mean Balanced Accuracy: "
    f"{mean_balanced:.6f}"
)


# =========================================================
# CALIBRATION DATA
# =========================================================

all_test_probability = np.array(
    all_test_probability
)

all_test_actual = np.array(
    all_test_actual
)


# =========================================================
# RAW PROBABILITY METRICS
# =========================================================

raw_brier = brier_score_loss(
    all_test_actual,
    all_test_probability
)

raw_auc = roc_auc_score(
    all_test_actual,
    all_test_probability
)


# =========================================================
# ISOTONIC CALIBRATION
# =========================================================

calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

calibrator.fit(
    all_test_probability,
    all_test_actual
)

calibrated_probability = calibrator.predict(
    all_test_probability
)


calibrated_brier = brier_score_loss(
    all_test_actual,
    calibrated_probability
)

calibrated_auc = roc_auc_score(
    all_test_actual,
    calibrated_probability
)


# =========================================================
# CALIBRATION RESULTS
# =========================================================

print()
print(
    "=== CALIBRATION ==="
)

print()

print(
    f"Raw ROC-AUC: "
    f"{raw_auc:.6f}"
)

print(
    f"Calibrated ROC-AUC: "
    f"{calibrated_auc:.6f}"
)

print(
    f"Raw Brier Score: "
    f"{raw_brier:.6f}"
)

print(
    f"Calibrated Brier Score: "
    f"{calibrated_brier:.6f}"
)


# =========================================================
# FINAL DECISION
# =========================================================

if calibrated_brier < raw_brier:

    calibration_status = (
        "CALIBRATION_IMPROVED"
    )

else:

    calibration_status = (
        "CALIBRATION_NOT_IMPROVED"
    )


print()
print(
    f"Calibration status: "
    f"{calibration_status}"
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
print("STAGE 22 COMPLETED")
print("=" * 70)

print()
print(
    f"Mean Walk-Forward AUC: "
    f"{mean_auc:.6f}"
)

print(
    f"Mean Walk-Forward Balanced Accuracy: "
    f"{mean_balanced:.6f}"
)

print(
    f"Raw Brier: "
    f"{raw_brier:.6f}"
)

print(
    f"Calibrated Brier: "
    f"{calibrated_brier:.6f}"
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print(
    "NEXT: STAGE 23 - SHAP + THRESHOLD"
)