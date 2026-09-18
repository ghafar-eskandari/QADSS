import pandas as pd

from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

INPUT_PATH = "data/processed/webmelat_model_ready.csv"

FEATURES = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_20",
    "price_to_ma20",
    "volatility_20",
    "volume",
    "no_trade",
]

TARGET = "target"

PURGE_DAYS = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Three-way temporal split
# --------------------------------------------------

n = len(df)

train_end = int(n * 0.50)

calibration_start = train_end + PURGE_DAYS

calibration_end = int(n * 0.60)

test_start = calibration_end + PURGE_DAYS

train = df.iloc[:train_end].copy()

calibration = df.iloc[calibration_start:calibration_end].copy()

test = df.iloc[test_start:].copy()


# --------------------------------------------------
# Prepare data
# --------------------------------------------------

X_train = train[FEATURES]
y_train = train[TARGET]

X_calibration = calibration[FEATURES]
y_calibration = calibration[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# --------------------------------------------------
# XGBoost base model
# --------------------------------------------------

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


# --------------------------------------------------
# Raw probabilities
# --------------------------------------------------

calibration_prob = model.predict_proba(
    X_calibration
)[:, 1]

test_prob_raw = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# Isotonic calibration
# --------------------------------------------------

calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

calibrator.fit(
    calibration_prob,
    y_calibration
)

test_prob_calibrated = calibrator.predict(
    test_prob_raw
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

raw_auc = roc_auc_score(
    y_test,
    test_prob_raw
)

calibrated_auc = roc_auc_score(
    y_test,
    test_prob_calibrated
)

raw_brier = brier_score_loss(
    y_test,
    test_prob_raw
)

calibrated_brier = brier_score_loss(
    y_test,
    test_prob_calibrated
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("Time-based probability calibration completed.")
print()

print("Train rows:", len(train))
print("Calibration rows:", len(calibration))
print("Test rows:", len(test))
print()

print("Train period:")
print(
    train["date"].min().date(),
    "to",
    train["date"].max().date()
)

print()

print("Calibration period:")
print(
    calibration["date"].min().date(),
    "to",
    calibration["date"].max().date()
)

print()

print("Test period:")
print(
    test["date"].min().date(),
    "to",
    test["date"].max().date()
)

print()

print("Raw XGBoost ROC-AUC:", round(raw_auc, 4))
print("Calibrated ROC-AUC:", round(calibrated_auc, 4))
print()

print("Raw Brier score:", round(raw_brier, 4))
print("Calibrated Brier score:", round(calibrated_brier, 4))
print()

print("Raw probability summary:")
print(
    pd.Series(test_prob_raw).describe()
)

print()

print("Calibrated probability summary:")
print(
    pd.Series(test_prob_calibrated).describe()
)

print()

print("First 10 raw probabilities:")
print(
    pd.Series(test_prob_raw[:10]).round(4).to_string(index=False)
)

print()

print("First 10 calibrated probabilities:")
print(
    pd.Series(test_prob_calibrated[:10])
    .round(4)
    .to_string(index=False)
)