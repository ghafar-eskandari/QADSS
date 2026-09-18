import pandas as pd
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
    "training_size_controlled_results.csv"
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

        return pd.to_datetime(
            value,
            errors="coerce"
        )

    value = value.replace(".0", "")

    if len(value) == 8:

        return pd.to_datetime(
            value,
            format="%Y%m%d",
            errors="coerce"
        )

    return pd.to_datetime(
        value,
        errors="coerce"
    )


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
# 3. IDENTIFY STRUCTURAL EVENTS
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
# 4. ORIGINAL TEST START
# =========================================================

split_idx = int(
    len(original) * 0.60
)

original_test_start = (
    original.iloc[split_idx]["date_key"]
)


# =========================================================
# 5. COMMON TEST DATES
# =========================================================

original_test_dates = set(
    original.loc[
        original["date_key"] >= original_test_start,
        "date_key"
    ]
)

clean_test_dates = set(
    clean.loc[
        clean["date_key"] >= original_test_start,
        "date_key"
    ]
)

common_test_dates = sorted(
    original_test_dates
    .intersection(clean_test_dates)
)

if len(common_test_dates) == 0:

    raise ValueError(
        "No common test dates found."
    )


common_test_start = min(
    common_test_dates
)

common_test_end = max(
    common_test_dates
)


# =========================================================
# 6. COMMON TEST DATA
# =========================================================

original_test = original[
    original["date_key"].isin(
        common_test_dates
    )
].copy()

clean_test = clean[
    clean["date_key"].isin(
        common_test_dates
    )
].copy()


# =========================================================
# 7. CREATE TRAINING DATA
# =========================================================

original_train_full = original[
    original["date_key"] < common_test_start
].copy()

clean_train_full = clean[
    clean["date_key"] < common_test_start
].copy()


# =========================================================
# 8. APPLY PURGE=5
# =========================================================

if len(original_train_full) <= 5:
    raise ValueError(
        "Original training data insufficient."
    )

if len(clean_train_full) <= 5:
    raise ValueError(
        "Clean training data insufficient."
    )


original_train_full = (
    original_train_full
    .sort_values("date_key")
    .reset_index(drop=True)
)

clean_train_full = (
    clean_train_full
    .sort_values("date_key")
    .reset_index(drop=True)
)


original_train_purged = (
    original_train_full
    .iloc[:-5]
    .copy()
)

clean_train_purged = (
    clean_train_full
    .iloc[:-5]
    .copy()
)


# =========================================================
# 9. CONTROL TRAINING SIZE
# =========================================================

clean_train_size = len(
    clean_train_purged
)

if len(original_train_purged) < clean_train_size:

    raise ValueError(
        "Original training dataset is smaller "
        "than clean training dataset."
    )


# ---------------------------------------------------------
# Use the most recent N ORIGINAL observations.
# This keeps the training period as comparable as
# possible to the clean dataset.
# ---------------------------------------------------------

original_train_controlled = (
    original_train_purged
    .tail(clean_train_size)
    .copy()
)


# =========================================================
# 10. REPORT TRAINING PERIODS
# =========================================================

print()
print(
    "=== STAGE 20.17: "
    "TRAINING-SIZE CONTROLLED SENSITIVITY ==="
)
print()

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
    "=== TRAINING DATA ==="
)
print()

print(
    "ORIGINAL FULL TRAIN:"
)

print(
    f"Rows: "
    f"{len(original_train_purged)}"
)

print(
    f"Period: "
    f"{original_train_purged['date_key'].min().date()} "
    f"-> "
    f"{original_train_purged['date_key'].max().date()}"
)

print()
print(
    "EVENT-AWARE CLEAN TRAIN:"
)

print(
    f"Rows: "
    f"{len(clean_train_purged)}"
)

print(
    f"Period: "
    f"{clean_train_purged['date_key'].min().date()} "
    f"-> "
    f"{clean_train_purged['date_key'].max().date()}"
)

print()
print(
    "ORIGINAL SIZE-CONTROLLED TRAIN:"
)

print(
    f"Rows: "
    f"{len(original_train_controlled)}"
)

print(
    f"Period: "
    f"{original_train_controlled['date_key'].min().date()} "
    f"-> "
    f"{original_train_controlled['date_key'].max().date()}"
)


# =========================================================
# 11. MODEL EVALUATION
# =========================================================

def evaluate_model(
    train_df,
    test_df,
    dataset_name
):

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "target"
    ].astype(int)

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "target"
    ].astype(int)

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

    prediction = (
        probability >= 0.50
    ).astype(int)

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

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "roc_auc": roc_auc,

        "tn": int(cm[0, 0]),

        "fp": int(cm[0, 1]),

        "fn": int(cm[1, 0]),

        "tp": int(cm[1, 1])
    }


# =========================================================
# 12. RUN CONTROLLED COMPARISON
# =========================================================

results = []

# ---------------------------------------------------------
# A. Original with controlled training size
# ---------------------------------------------------------

results.append(
    evaluate_model(
        original_train_controlled,
        original_test,
        "ORIGINAL_SIZE_CONTROLLED"
    )
)


# ---------------------------------------------------------
# B. Event-aware clean
# ---------------------------------------------------------

results.append(
    evaluate_model(
        clean_train_purged,
        clean_test,
        "EVENT_AWARE_CLEAN"
    )
)


results_df = pd.DataFrame(
    results
)


# =========================================================
# 13. MODEL COMPARISON
# =========================================================

print()
print(
    "=== TRAINING-SIZE CONTROLLED "
    "MODEL COMPARISON ==="
)
print()

print(
    results_df[
        [
            "dataset",
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
    ].to_string(index=False)
)


# =========================================================
# 14. CONFUSION MATRICES
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
# 15. DELTA ANALYSIS
# =========================================================

original_auc = float(
    results_df.loc[
        results_df["dataset"] ==
        "ORIGINAL_SIZE_CONTROLLED",
        "roc_auc"
    ].iloc[0]
)

clean_auc = float(
    results_df.loc[
        results_df["dataset"] ==
        "EVENT_AWARE_CLEAN",
        "roc_auc"
    ].iloc[0]
)

auc_delta = (
    clean_auc -
    original_auc
)


original_balanced = float(
    results_df.loc[
        results_df["dataset"] ==
        "ORIGINAL_SIZE_CONTROLLED",
        "balanced_accuracy"
    ].iloc[0]
)

clean_balanced = float(
    results_df.loc[
        results_df["dataset"] ==
        "EVENT_AWARE_CLEAN",
        "balanced_accuracy"
    ].iloc[0]
)

balanced_delta = (
    clean_balanced -
    original_balanced
)


print()
print(
    "=== CONTROLLED DIFFERENCE ==="
)
print()

print(
    f"ROC-AUC difference "
    f"(Clean - Original controlled): "
    f"{auc_delta:.6f}"
)

print(
    f"Balanced Accuracy difference "
    f"(Clean - Original controlled): "
    f"{balanced_delta:.6f}"
)


# =========================================================
# 16. SAVE RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 17. FINAL SUMMARY
# =========================================================

print()
print(
    "=== STAGE 20.17 SUMMARY ==="
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
    "Both scenarios use exactly the same "
    "common Test dates."
)

print(
    "Both scenarios use the same number "
    "of training observations."
)

print(
    "The Original controlled dataset uses "
    "the most recent training observations "
    "available before the common Test period."
)

print(
    "No data was permanently deleted."
)

print(
    "No final dataset was selected."
)

print(
    "This stage isolates the effect of "
    "event-aware cleaning from the effect "
    "of training-set size."
)