import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)


# ============================================================
# QADSS - FINAL CALIBRATION - V2 XGBOOST
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "webmelat_model_ready_v2.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "v2_calibration_results.csv"
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


print("=" * 80)
print("QADSS - FINAL CALIBRATION - V2 XGBOOST")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("\n")
print(f"Total observations: {len(df)}")


# ============================================================
# 2. PREPARE DATA
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].astype(int)


# ============================================================
# 3. TIME-BASED SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.50)
calibration_end = int(n * 0.60)

train_start = 0
calibration_start = train_end + PURGE
test_start = calibration_end + PURGE

X_train = X.iloc[
    train_start:train_end
]

y_train = y.iloc[
    train_start:train_end
]

X_calibration = X.iloc[
    calibration_start:calibration_end
]

y_calibration = y.iloc[
    calibration_start:calibration_end
]

X_test = X.iloc[
    test_start:
]

y_test = y.iloc[
    test_start:
]


print("\n")
print("=" * 80)
print("TIME-BASED SPLIT")
print("=" * 80)

print(
    f"Training observations   : {len(X_train)}"
)

print(
    f"Calibration observations: {len(X_calibration)}"
)

print(
    f"Test observations       : {len(X_test)}"
)

print(
    f"Purged observations     : {PURGE}"
)


# ============================================================
# 4. TRAIN XGBOOST
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
# 5. RAW PROBABILITIES
# ============================================================

p_calibration_raw = model.predict_proba(
    X_calibration
)[:, 1]

p_test_raw = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. ISOTONIC CALIBRATION
# ============================================================

calibrator = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip"
)

calibrator.fit(
    p_calibration_raw,
    y_calibration
)


p_test_calibrated = calibrator.predict(
    p_test_raw
)


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

raw_auc = roc_auc_score(
    y_test,
    p_test_raw
)

calibrated_auc = roc_auc_score(
    y_test,
    p_test_calibrated
)

raw_brier = brier_score_loss(
    y_test,
    p_test_raw
)

calibrated_brier = brier_score_loss(
    y_test,
    p_test_calibrated
)


# ============================================================
# 8. PROBABILITY SUMMARY
# ============================================================

raw_mean = np.mean(
    p_test_raw
)

raw_std = np.std(
    p_test_raw
)

calibrated_mean = np.mean(
    p_test_calibrated
)

calibrated_std = np.std(
    p_test_calibrated
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("CALIBRATION RESULTS")
print("=" * 80)

print(
    f"\nRaw XGBoost AUC       : "
    f"{raw_auc:.4f}"
)

print(
    f"Calibrated XGBoost AUC: "
    f"{calibrated_auc:.4f}"
)

print(
    f"\nRaw Brier Score       : "
    f"{raw_brier:.4f}"
)

print(
    f"Calibrated Brier Score: "
    f"{calibrated_brier:.4f}"
)


print("\n")
print("=" * 80)
print("PROBABILITY DISTRIBUTION")
print("=" * 80)

print(
    f"\nRaw probability:"
)

print(
    f"  Mean : {raw_mean:.6f}"
)

print(
    f"  Std  : {raw_std:.6f}"
)

print(
    f"  Min  : {p_test_raw.min():.6f}"
)

print(
    f"  Max  : {p_test_raw.max():.6f}"
)


print(
    f"\nCalibrated probability:"
)

print(
    f"  Mean : {calibrated_mean:.6f}"
)

print(
    f"  Std  : {calibrated_std:.6f}"
)

print(
    f"  Min  : {p_test_calibrated.min():.6f}"
)

print(
    f"  Max  : {p_test_calibrated.max():.6f}"
)


# ============================================================
# 10. INTERPRETATION
# ============================================================

print("\n")
print("=" * 80)
print("CALIBRATION INTERPRETATION")
print("=" * 80)

if calibrated_brier < raw_brier:

    print(
        "\nCalibration improved probability quality "
        "according to Brier Score."
    )

else:

    print(
        "\nCalibration did NOT improve Brier Score."
    )


if calibrated_auc > raw_auc:

    print(
        "Calibration increased ROC-AUC."
    )

elif calibrated_auc < raw_auc:

    print(
        "Calibration decreased ROC-AUC."
    )

else:

    print(
        "Calibration did not change ROC-AUC."
    )


print(
    "\nImportant:"
)

print(
    "Calibration changes probability quality; "
    "it does not automatically create predictive power."
)


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

results = pd.DataFrame(
    [
        {
            "Model": "V2 XGBoost",
            "Raw AUC": raw_auc,
            "Calibrated AUC": calibrated_auc,
            "Raw Brier": raw_brier,
            "Calibrated Brier": calibrated_brier,
            "Raw Probability Mean": raw_mean,
            "Raw Probability Std": raw_std,
            "Calibrated Probability Mean": calibrated_mean,
            "Calibrated Probability Std": calibrated_std,
        }
    ]
)


results.to_csv(
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