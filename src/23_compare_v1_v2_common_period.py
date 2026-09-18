import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# QADSS - V1 vs V2
# COMMON DATE PERIOD COMPARISON
# ============================================================

V1_FILE = "data/processed/webmelat_model_ready.csv"
V2_FILE = "data/processed/webmelat_model_ready_v2.csv"

PURGE_DAYS = 5
TRAIN_RATIO = 0.40
TEST_RATIO = 0.10


FEATURES_V1 = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_20",
    "price_to_ma20",
    "volatility_20",
    "volume",
    "no_trade",
]

FEATURES_V2 = [
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


# ============================================================
# LOAD DATA
# ============================================================

v1 = pd.read_csv(V1_FILE)
v2 = pd.read_csv(V2_FILE)

print("=" * 80)
print("QADSS - V1 vs V2 COMMON DATE PERIOD COMPARISON")
print("=" * 80)

print(f"V1 observations: {len(v1)}")
print(f"V2 observations: {len(v2)}")


# ============================================================
# CHECK DATE COLUMN
# ============================================================

if "dEven" not in v1.columns:
    raise ValueError(
        "V1 does not contain dEven."
    )

if "dEven" not in v2.columns:
    raise ValueError(
        "V2 does not contain dEven."
    )


# ============================================================
# CONVERT DATE
# ============================================================

# ============================================================
# CONVERT DATE TO STANDARD DATETIME
# ============================================================

v1["dEven"] = pd.to_datetime(
    v1["dEven"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

v2["dEven"] = pd.to_datetime(
    v2["dEven"].astype(str),
    format="%Y-%m-%d",
    errors="coerce"
)

if v1["dEven"].isna().any():
    raise ValueError(
        "Invalid dates found in V1."
    )

if v2["dEven"].isna().any():
    raise ValueError(
        "Invalid dates found in V2."
    )

# ============================================================
# ALIGN BY DATE
# ============================================================

common_dates = sorted(
    set(v1["dEven"]).intersection(
        set(v2["dEven"])
    )
)

print(
    f"Common trading dates: {len(common_dates)}"
)

if len(common_dates) == 0:
    raise ValueError(
        "No common trading dates found."
    )


v1 = (
    v1[v1["dEven"].isin(common_dates)]
    .sort_values("dEven")
    .reset_index(drop=True)
)

v2 = (
    v2[v2["dEven"].isin(common_dates)]
    .sort_values("dEven")
    .reset_index(drop=True)
)


# ============================================================
# FINAL DATE ALIGNMENT CHECK
# ============================================================

if not np.array_equal(
    v1["dEven"].values,
    v2["dEven"].values
):
    raise ValueError(
        "Date alignment failed."
    )

print("Date alignment: OK")

print(
    f"Common period: "
    f"{common_dates[0]} -> {common_dates[-1]}"
)


# ============================================================
# TARGET ALIGNMENT CHECK
# ============================================================

if not np.array_equal(
    v1["target"].astype(int).values,
    v2["target"].astype(int).values
):
    raise ValueError(
        "Target mismatch after date alignment."
    )

print("Target alignment: OK")


# ============================================================
# NUMBER OF OBSERVATIONS
# ============================================================

common_n = len(v1)

print(
    f"Common-period observations: {common_n}"
)


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

initial_train_size = int(
    common_n * TRAIN_RATIO
)

test_size = int(
    common_n * TEST_RATIO
)

print(
    f"Initial training size: "
    f"{initial_train_size}"
)

print(
    f"Test size per fold: "
    f"{test_size}"
)

print(
    f"Purged rows: "
    f"{PURGE_DAYS}"
)


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_logistic():

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]
    )


def create_xgb():

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


# ============================================================
# RESULTS
# ============================================================

results = []

train_end = initial_train_size
fold = 1


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

while True:

    test_start = train_end + PURGE_DAYS
    test_end = test_start + test_size

    if test_end > common_n:
        break


    print("\n" + "=" * 80)
    print(f"FOLD {fold}")
    print("=" * 80)

    print(
        f"Train: 0 -> {train_end - 1}"
    )

    print(
        f"Purge: {train_end} -> "
        f"{test_start - 1}"
    )

    print(
        f"Test: {test_start} -> "
        f"{test_end - 1}"
    )


    print(
        f"Test dates: "
        f"{v1.loc[test_start, 'dEven']} -> "
        f"{v1.loc[test_end - 1, 'dEven']}"
    )


    # ========================================================
    # V1
    # ========================================================

    X1_train = v1.loc[
        :train_end - 1,
        FEATURES_V1
    ]

    y1_train = v1.loc[
        :train_end - 1,
        "target"
    ].astype(int)

    X1_test = v1.loc[
        test_start:test_end - 1,
        FEATURES_V1
    ]

    y1_test = v1.loc[
        test_start:test_end - 1,
        "target"
    ].astype(int)


    # ========================================================
    # V2
    # ========================================================

    X2_train = v2.loc[
        :train_end - 1,
        FEATURES_V2
    ]

    y2_train = v2.loc[
        :train_end - 1,
        "target"
    ].astype(int)

    X2_test = v2.loc[
        test_start:test_end - 1,
        FEATURES_V2
    ]

    y2_test = v2.loc[
        test_start:test_end - 1,
        "target"
    ].astype(int)


    # ========================================================
    # V1 LOGISTIC
    # ========================================================

    model = create_logistic()

    model.fit(X1_train, y1_train)

    pred = model.predict(X1_test)
    prob = model.predict_proba(X1_test)[:, 1]

    v1_log_acc = accuracy_score(
        y1_test,
        pred
    )

    v1_log_bal = balanced_accuracy_score(
        y1_test,
        pred
    )

    v1_log_auc = roc_auc_score(
        y1_test,
        prob
    )


    # ========================================================
    # V1 XGBOOST
    # ========================================================

    model = create_xgb()

    model.fit(X1_train, y1_train)

    pred = model.predict(X1_test)
    prob = model.predict_proba(X1_test)[:, 1]

    v1_xgb_acc = accuracy_score(
        y1_test,
        pred
    )

    v1_xgb_bal = balanced_accuracy_score(
        y1_test,
        pred
    )

    v1_xgb_auc = roc_auc_score(
        y1_test,
        prob
    )


    # ========================================================
    # V2 LOGISTIC
    # ========================================================

    model = create_logistic()

    model.fit(X2_train, y2_train)

    pred = model.predict(X2_test)
    prob = model.predict_proba(X2_test)[:, 1]

    v2_log_acc = accuracy_score(
        y2_test,
        pred
    )

    v2_log_bal = balanced_accuracy_score(
        y2_test,
        pred
    )

    v2_log_auc = roc_auc_score(
        y2_test,
        prob
    )


    # ========================================================
    # V2 XGBOOST
    # ========================================================

    model = create_xgb()

    model.fit(X2_train, y2_train)

    pred = model.predict(X2_test)
    prob = model.predict_proba(X2_test)[:, 1]

    v2_xgb_acc = accuracy_score(
        y2_test,
        pred
    )

    v2_xgb_bal = balanced_accuracy_score(
        y2_test,
        pred
    )

    v2_xgb_auc = roc_auc_score(
        y2_test,
        prob
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\nV1 Logistic:")
    print(
        f"Accuracy={v1_log_acc:.4f}, "
        f"Balanced={v1_log_bal:.4f}, "
        f"AUC={v1_log_auc:.4f}"
    )

    print("\nV1 XGBoost:")
    print(
        f"Accuracy={v1_xgb_acc:.4f}, "
        f"Balanced={v1_xgb_bal:.4f}, "
        f"AUC={v1_xgb_auc:.4f}"
    )

    print("\nV2 Logistic:")
    print(
        f"Accuracy={v2_log_acc:.4f}, "
        f"Balanced={v2_log_bal:.4f}, "
        f"AUC={v2_log_auc:.4f}"
    )

    print("\nV2 XGBoost:")
    print(
        f"Accuracy={v2_xgb_acc:.4f}, "
        f"Balanced={v2_xgb_bal:.4f}, "
        f"AUC={v2_xgb_auc:.4f}"
    )


    # ========================================================
    # SAVE
    # ========================================================

    results.append(
        {
            "fold": fold,

            "v1_logistic_accuracy": v1_log_acc,
            "v1_logistic_balanced": v1_log_bal,
            "v1_logistic_auc": v1_log_auc,

            "v1_xgb_accuracy": v1_xgb_acc,
            "v1_xgb_balanced": v1_xgb_bal,
            "v1_xgb_auc": v1_xgb_auc,

            "v2_logistic_accuracy": v2_log_acc,
            "v2_logistic_balanced": v2_log_bal,
            "v2_logistic_auc": v2_log_auc,

            "v2_xgb_accuracy": v2_xgb_acc,
            "v2_xgb_balanced": v2_xgb_bal,
            "v2_xgb_auc": v2_xgb_auc,
        }
    )


    train_end = test_end
    fold += 1


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("COMMON-DATE WALK-FORWARD SUMMARY")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# MEAN PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("MEAN PERFORMANCE")
print("=" * 80)

summary = {
    "V1 Logistic": [
        results_df["v1_logistic_accuracy"].mean(),
        results_df["v1_logistic_balanced"].mean(),
        results_df["v1_logistic_auc"].mean(),
    ],

    "V1 XGBoost": [
        results_df["v1_xgb_accuracy"].mean(),
        results_df["v1_xgb_balanced"].mean(),
        results_df["v1_xgb_auc"].mean(),
    ],

    "V2 Logistic": [
        results_df["v2_logistic_accuracy"].mean(),
        results_df["v2_logistic_balanced"].mean(),
        results_df["v2_logistic_auc"].mean(),
    ],

    "V2 XGBoost": [
        results_df["v2_xgb_accuracy"].mean(),
        results_df["v2_xgb_balanced"].mean(),
        results_df["v2_xgb_auc"].mean(),
    ],
}


summary_df = pd.DataFrame(
    summary,
    index=[
        "Mean Accuracy",
        "Mean Balanced Accuracy",
        "Mean ROC-AUC",
    ]
)

print(
    summary_df.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    "data/processed/"
    "v1_v2_common_period_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 80)