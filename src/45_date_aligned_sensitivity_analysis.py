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
    "date_aligned_sensitivity_results.csv"
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
# 3. IDENTIFY STRUCTURAL-BREAK EVENTS
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
# 4. STRUCTURAL-BREAK EXCLUSION WINDOW
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
# 5. ORIGINAL DATA DEFINES FIXED TIME BOUNDARY
# =========================================================

original_start = (
    original["date_key"].min()
)

original_end = (
    original["date_key"].max()
)

split_idx = int(
    len(original) * 0.60
)

test_start_date = (
    original.iloc[split_idx]["date_key"]
)


# =========================================================
# 6. REPORT STAGE INFORMATION
# =========================================================

print()
print(
    "=== STAGE 20.15: "
    "DATE-ALIGNED SENSITIVITY ANALYSIS ==="
)
print()

print(
    f"Original rows: {len(original)}"
)

print(
    f"Event-aware clean rows: {len(clean)}"
)

print(
    f"Structural-break excluded rows: "
    f"{len(structural_clean)}"
)

print()
print(
    "=== FIXED TEMPORAL BOUNDARIES ==="
)
print()

print(
    f"Original full period: "
    f"{original_start.date()} -> "
    f"{original_end.date()}"
)

print(
    f"Test starts from: "
    f"{test_start_date.date()}"
)

print(
    f"Structural-break events: "
    f"{len(structural_events)}"
)

for event_date in structural_events:

    print(
        f"  - {event_date.date()}"
    )


# =========================================================
# 7. MODEL EVALUATION FUNCTION
# =========================================================

def evaluate_dataset(
    df,
    dataset_name
):

    df = df.copy()

    # -----------------------------------------------------
    # Keep exactly the same overall temporal range
    # -----------------------------------------------------

    df = df[
        (df["date_key"] >= original_start) &
        (df["date_key"] <= original_end)
    ].copy()

    df = (
        df
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Fixed temporal split
    # -----------------------------------------------------

    train = df[
        df["date_key"] < test_start_date
    ].copy()

    test = df[
        df["date_key"] >= test_start_date
    ].copy()

    # -----------------------------------------------------
    # Purge = 5 observations
    # -----------------------------------------------------

    if len(train) <= 5:

        raise ValueError(
            f"{dataset_name}: "
            "Not enough training observations "
            "for purge=5."
        )

    train = train.iloc[:-5].copy()

    # -----------------------------------------------------
    # Features / target
    # -----------------------------------------------------

    X_train = train[FEATURES].copy()
    y_train = train["target"].astype(int)

    X_test = test[FEATURES].copy()
    y_test = test["target"].astype(int)

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

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    return {

        "dataset": dataset_name,

        "rows": len(df),

        "train_rows": len(train),

        "test_rows": len(test),

        "train_start": (
            train["date_key"]
            .min()
            .strftime("%Y-%m-%d")
        ),

        "train_end": (
            train["date_key"]
            .max()
            .strftime("%Y-%m-%d")
        ),

        "test_start": (
            test["date_key"]
            .min()
            .strftime("%Y-%m-%d")
        ),

        "test_end": (
            test["date_key"]
            .max()
            .strftime("%Y-%m-%d")
        ),

        "target_0": int(
            (df["target"] == 0).sum()
        ),

        "target_1": int(
            (df["target"] == 1).sum()
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
# 8. RUN THREE SCENARIOS
# =========================================================

results = []

results.append(
    evaluate_dataset(
        original,
        "ORIGINAL"
    )
)

results.append(
    evaluate_dataset(
        clean,
        "EVENT_AWARE_CLEAN"
    )
)

results.append(
    evaluate_dataset(
        structural_clean,
        "STRUCTURAL_BREAK_EXCLUDED"
    )
)


results_df = pd.DataFrame(
    results
)


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

print()
print(
    "=== DATE-ALIGNED MODEL COMPARISON ==="
)
print()

comparison_columns = [

    "dataset",

    "rows",

    "train_rows",

    "test_rows",

    "train_start",

    "train_end",

    "test_start",

    "test_end",

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
# 10. CONFUSION MATRICES
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
# 11. SAVE RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

print()
print(
    "=== STAGE 20.15 SUMMARY ==="
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
    "All three scenarios use the same "
    "temporal Train/Test boundary "
    "defined by ORIGINAL."
)

print(
    "No data was permanently deleted."
)

print(
    "No final dataset was selected."
)

print(
    "This stage evaluates whether the "
    "sensitivity observed in Stage 20.14 "
    "remains after temporal alignment."
)