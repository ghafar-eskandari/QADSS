import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[1]

ORIGINAL_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2.csv"
)

CLEAN_FILE = (
    BASE / "data" / "processed" /
    "webmelat_model_ready_v2_clean.csv"
)

EVENT_FLAG_FILE = (
    BASE / "data" / "processed" /
    "final_event_flags.csv"
)

OUTPUT_FILE = (
    BASE / "data" / "processed" /
    "common_test_period_sensitivity_results.csv"
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


# =========================================================
# 1. DATE NORMALIZATION
# =========================================================

def normalize_date(value):

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if "-" in value:

        dt = pd.to_datetime(
            value,
            errors="coerce"
        )

    else:

        value = value.replace(".0", "")

        if len(value) == 8:

            dt = pd.to_datetime(
                value,
                format="%Y%m%d",
                errors="coerce"
            )

        else:

            dt = pd.to_datetime(
                value,
                errors="coerce"
            )

    return dt


# =========================================================
# 2. LOAD DATA
# =========================================================

original = pd.read_csv(
    ORIGINAL_FILE
)

clean = pd.read_csv(
    CLEAN_FILE
)

flags = pd.read_csv(
    EVENT_FLAG_FILE
)


original["date_key"] = (
    original["dEven"]
    .apply(normalize_date)
)

clean["date_key"] = (
    clean["dEven"]
    .apply(normalize_date)
)

flags["date_key"] = (
    flags["event_date"]
    .apply(normalize_date)
)


original = (
    original
    .dropna(subset=["date_key"])
    .sort_values("date_key")
    .reset_index(drop=True)
)

clean = (
    clean
    .dropna(subset=["date_key"])
    .sort_values("date_key")
    .reset_index(drop=True)
)

flags = (
    flags
    .dropna(subset=["date_key"])
    .copy()
)


# =========================================================
# 3. STRUCTURAL-BREAK EVENTS
# =========================================================

structural_events = (
    flags.loc[
        flags["structural_break_flag"] == 1,
        "date_key"
    ]
    .drop_duplicates()
    .tolist()
)


# =========================================================
# 4. STRUCTURAL-BREAK EXCLUSION
# =========================================================

structural_dates = set()

for event_date in structural_events:

    for offset in range(-20, 21):

        affected_date = (
            event_date +
            pd.Timedelta(days=offset)
        ).date()

        structural_dates.add(
            affected_date
        )


structural_clean = original[
    ~original["date_key"].dt.date.isin(
        structural_dates
    )
].copy()


# =========================================================
# 5. ORIGINAL TEMPORAL SPLIT
# =========================================================

original = (
    original
    .sort_values("date_key")
    .reset_index(drop=True)
)

split_idx = int(
    len(original) * 0.60
)

original_test_start = (
    original.iloc[split_idx]["date_key"]
)


# =========================================================
# 6. DEFINE TEST PERIOD IN ORIGINAL DATA
# =========================================================

original_test = original[
    original["date_key"] >= original_test_start
].copy()

original_test_dates = set(
    original_test["date_key"]
)


# =========================================================
# 7. FIND COMMON TEST DATES
# =========================================================

clean_test_dates = set(
    clean.loc[
        clean["date_key"] >= original_test_start,
        "date_key"
    ]
)

structural_test_dates = set(
    structural_clean.loc[
        structural_clean["date_key"] >= original_test_start,
        "date_key"
    ]
)

common_test_dates = (
    original_test_dates
    .intersection(clean_test_dates)
    .intersection(structural_test_dates)
)

common_test_dates = sorted(
    common_test_dates
)


if len(common_test_dates) == 0:

    raise ValueError(
        "No common test dates were found."
    )


common_test_start = (
    min(common_test_dates)
)

common_test_end = (
    max(common_test_dates)
)


# =========================================================
# 8. COMMON TEST DATASETS
# =========================================================

original_common_test = original[
    original["date_key"].isin(
        common_test_dates
    )
].copy()

clean_common_test = clean[
    clean["date_key"].isin(
        common_test_dates
    )
].copy()

structural_common_test = structural_clean[
    structural_clean["date_key"].isin(
        common_test_dates
    )
].copy()


# =========================================================
# 9. TRAIN DATA
# =========================================================

def create_train_data(df):

    train = df[
        df["date_key"] < common_test_start
    ].copy()

    train = (
        train
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    if len(train) <= 5:

        raise ValueError(
            "Training data is insufficient "
            "for purge=5."
        )

    # Purge five observations immediately
    # before the common test period.

    train = train.iloc[:-5].copy()

    return train


# =========================================================
# 10. EVALUATION FUNCTION
# =========================================================

def evaluate_dataset(
    train_df,
    test_df,
    dataset_name
):

    train_df = (
        train_df
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    if len(train_df) == 0:

        raise ValueError(
            f"{dataset_name}: empty training dataset."
        )

    if len(test_df) == 0:

        raise ValueError(
            f"{dataset_name}: empty test dataset."
        )

    X_train = train_df[
        FEATURES
    ].copy()

    y_train = train_df[
        "target"
    ].astype(int)

    X_test = test_df[
        FEATURES
    ].copy()

    y_test = test_df[
        "target"
    ].astype(int)

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    probability = (
        model
        .predict_proba(X_test)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            prediction
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    return {

        "dataset": dataset_name,

        "train_rows": len(train_df),

        "test_rows": len(test_df),

        "train_start": (
            train_df["date_key"]
            .min()
            .strftime("%Y-%m-%d")
        ),

        "train_end": (
            train_df["date_key"]
            .max()
            .strftime("%Y-%m-%d")
        ),

        "test_start": (
            test_df["date_key"]
            .min()
            .strftime("%Y-%m-%d")
        ),

        "test_end": (
            test_df["date_key"]
            .max()
            .strftime("%Y-%m-%d")
        ),

        "target_0_test": int(
            (y_test == 0).sum()
        ),

        "target_1_test": int(
            (y_test == 1).sum()
        ),

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "roc_auc": roc_auc,

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp
    }


# =========================================================
# 11. CREATE TRAIN DATASETS
# =========================================================

original_train = create_train_data(
    original[
        original["date_key"] < common_test_start
    ].copy()
)

clean_train = create_train_data(
    clean[
        clean["date_key"] < common_test_start
    ].copy()
)

structural_train = create_train_data(
    structural_clean[
        structural_clean["date_key"] < common_test_start
    ].copy()
)


# =========================================================
# 12. RUN MODELS
# =========================================================

results = []

results.append(
    evaluate_dataset(
        original_train,
        original_common_test,
        "ORIGINAL"
    )
)

results.append(
    evaluate_dataset(
        clean_train,
        clean_common_test,
        "EVENT_AWARE_CLEAN"
    )
)

results.append(
    evaluate_dataset(
        structural_train,
        structural_common_test,
        "STRUCTURAL_BREAK_EXCLUDED"
    )
)


results_df = pd.DataFrame(
    results
)


# =========================================================
# 13. REPORT COMMON TEST PERIOD
# =========================================================

print()
print(
    "=== STAGE 20.16: "
    "COMMON TEST PERIOD SENSITIVITY ==="
)
print()

print(
    f"Original test start: "
    f"{original_test_start.date()}"
)

print(
    f"Common test start: "
    f"{common_test_start.date()}"
)

print(
    f"Common test end: "
    f"{common_test_end.date()}"
)

print(
    f"Common test dates: "
    f"{len(common_test_dates)}"
)

print()
print(
    "Structural-break events: "
    f"{len(structural_events)}"
)

for event_date in structural_events:

    print(
        f"  - {event_date.date()}"
    )


# =========================================================
# 14. COMMON TEST DATASET SIZES
# =========================================================

print()
print(
    "=== COMMON TEST DATASET SIZES ==="
)
print()

print(
    f"Original common test rows: "
    f"{len(original_common_test)}"
)

print(
    f"Event-aware common test rows: "
    f"{len(clean_common_test)}"
)

print(
    f"Structural-break common test rows: "
    f"{len(structural_common_test)}"
)


# =========================================================
# 15. MODEL COMPARISON
# =========================================================

print()
print(
    "=== COMMON TEST MODEL COMPARISON ==="
)
print()

comparison_columns = [

    "dataset",

    "train_rows",

    "test_rows",

    "train_start",

    "train_end",

    "test_start",

    "test_end",

    "target_0_test",

    "target_1_test",

    "accuracy",

    "balanced_accuracy",

    "roc_auc"
]

print(
    results_df[
        comparison_columns
    ].to_string(index=False)
)


# =========================================================
# 16. CONFUSION MATRICES
# =========================================================

print()
print(
    "=== CONFUSION MATRICES ==="
)

for _, row in results_df.iterrows():

    print()
    print(
        row["dataset"]
    )

    print(
        f"TN={row['tn']} "
        f"FP={row['fp']} "
        f"FN={row['fn']} "
        f"TP={row['tp']}"
    )


# =========================================================
# 17. SAVE RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 18. FINAL SUMMARY
# =========================================================

print()
print(
    "=== STAGE 20.16 SUMMARY ==="
)
print()

print(
    results_df[
        [
            "dataset",
            "accuracy",
            "balanced_accuracy",
            "roc_auc"
        ]
    ].to_string(index=False)
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "All three scenarios are evaluated "
    "on exactly the same common Test dates."
)

print(
    "Each scenario uses its own available "
    "training observations before the common "
    "Test period."
)

print(
    "Purge=5 is applied to every training dataset."
)

print(
    "No data was permanently deleted."
)

print(
    "No final dataset was selected."
)

print(
    "This stage isolates the effect of "
    "test-period composition from the "
    "event-related data cleaning effect."
)