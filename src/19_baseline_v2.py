from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "webmelat_model_ready_v2.csv"
)


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

TARGET = "target"

TRAIN_RATIO = 0.60
PURGE_DAYS = 5


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# 4. Basic validation
# --------------------------------------------------

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

if df[TARGET].isna().any():
    raise ValueError(
        "Target contains missing values."
    )


# --------------------------------------------------
# 5. Preserve chronological order
# --------------------------------------------------

if "dEven" in df.columns:
    df = df.sort_values(
        "dEven"
    ).reset_index(drop=True)


# --------------------------------------------------
# 6. Prepare X and y
# --------------------------------------------------

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# --------------------------------------------------
# 7. Purged time split
# --------------------------------------------------

n = len(df)

split_index = int(
    TRAIN_RATIO * n
)

train_end = (
    split_index - PURGE_DAYS
)

train_indices = range(
    0,
    train_end
)

test_indices = range(
    split_index,
    n
)

X_train = X.iloc[
    train_end * 0 : train_end
].copy()

y_train = y.iloc[
    train_end * 0 : train_end
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_test = y.iloc[
    split_index:
].copy()


# --------------------------------------------------
# 8. Baseline prediction
# --------------------------------------------------

majority_class = (
    y_train
    .value_counts()
    .idxmax()
)

y_pred = [
    majority_class
    for _ in range(len(y_test))
]


# --------------------------------------------------
# 9. Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# --------------------------------------------------
# 10. Report
# --------------------------------------------------

print("=" * 80)
print("QADSS - BASELINE V2")
print("=" * 80)

print(
    f"Total observations: {n}"
)

print(
    f"Split index: {split_index}"
)

print(
    f"Purged rows: {PURGE_DAYS}"
)

print(
    f"Training observations: "
    f"{len(X_train)}"
)

print(
    f"Test observations: "
    f"{len(X_test)}"
)

print("\n")

print(
    f"Majority class: "
    f"{majority_class}"
)

print("\n")

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print("\n")

print("Confusion Matrix:")
print(cm)

print("\n")

if "dEven" in df.columns:

    train_dates = df.iloc[
        0
    ]["dEven"]

    train_end_date = df.iloc[
        train_end - 1
    ]["dEven"]

    test_start_date = df.iloc[
        split_index
    ]["dEven"]

    test_end_date = df.iloc[
        -1
    ]["dEven"]

    print(
        f"Train period: "
        f"{train_end_date}"
    )

    print(
        f"Test period: "
        f"{test_start_date} "
        f"to {test_end_date}"
    )

print("=" * 80)