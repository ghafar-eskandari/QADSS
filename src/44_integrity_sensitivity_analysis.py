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


# ============================================================
# STAGE 20.14
# INTEGRITY SENSITIVITY ANALYSIS
# ============================================================

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
    "integrity_sensitivity_results.csv"
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


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

original = pd.read_csv(
    ORIGINAL_FILE
)

clean = pd.read_csv(
    CLEAN_FILE
)

flags = pd.read_csv(
    EVENT_FLAG_FILE
)


# ------------------------------------------------------------
# 2. Normalize dates
# ------------------------------------------------------------

def normalize_date(value):

    if pd.isna(value):
        return np.nan

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

    if pd.isna(dt):
        return np.nan

    return dt.strftime("%Y-%m-%d")


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


# ------------------------------------------------------------
# 3. Identify structural-break event
# ------------------------------------------------------------

structural_events = flags[
    flags["structural_break_flag"] == 1
][
    "date_key"
].dropna().tolist()


print()
print(
    "=== STAGE 20.14: "
    "INTEGRITY SENSITIVITY ANALYSIS ==="
)

print()

print(
    f"Original rows: {len(original)}"
)

print(
    f"Event-aware clean rows: {len(clean)}"
)

print(
    f"Structural-break events: "
    f"{len(structural_events)}"
)

print(
    "Structural-break dates:",
    structural_events
)


# ------------------------------------------------------------
# 4. Create structural-break exclusion window
# ------------------------------------------------------------

structural_dates = set()

for event_date in structural_events:

    event_dt = pd.to_datetime(
        event_date
    )

    for offset in range(-20, 21):

        affected_date = (
            event_dt
            + pd.Timedelta(days=offset)
        ).strftime("%Y-%m-%d")

        structural_dates.add(
            affected_date
        )


structural_clean = original[
    ~original["date_key"].isin(
        structural_dates
    )
].copy()


# ------------------------------------------------------------
# 5. Model evaluation function
# ------------------------------------------------------------

def evaluate_dataset(
    df,
    dataset_name
):

    df = df.copy()

    df = df.sort_values(
        "date_key"
    ).reset_index(drop=True)


    X = df[FEATURES].copy()

    y = df["target"].astype(int)


    # --------------------------------------------------------
    # Fixed temporal split
    # --------------------------------------------------------

    split_idx = int(
        len(df) * 0.60
    )

    purge = 5

    train_end = split_idx - purge

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

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


    prob = model.predict_proba(
        X_test
    )[:, 1]

    pred = (
        prob >= 0.50
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        pred
    )

    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    auc = roc_auc_score(
        y_test,
        prob
    )

    cm = confusion_matrix(
        y_test,
        pred
    )


    return {
        "dataset": dataset_name,
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "target_0": int((y == 0).sum()),
        "target_1": int((y == 1).sum()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# ------------------------------------------------------------
# 6. Evaluate all three datasets
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. Display results
# ------------------------------------------------------------

print()
print(
    "=== MODEL COMPARISON ==="
)

print(
    results_df[
        [
            "dataset",
            "rows",
            "train_rows",
            "test_rows",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
        ]
    ].to_string(index=False)
)


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


# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------

print()
print(
    "=== STAGE 20.14 SUMMARY ==="
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
print("IMPORTANT:")
print("No data was permanently deleted.")
print("No final model was selected.")
print(
    "This stage measures sensitivity of model performance "
    "to event-related data cleaning."
)