import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)

from xgboost import XGBClassifier


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
# Walk-Forward settings
# --------------------------------------------------

n = len(df)

initial_train_size = int(n * 0.40)

test_size = int(n * 0.10)

results = []


# --------------------------------------------------
# Walk-Forward loop
# --------------------------------------------------

fold = 1

train_start = 0
train_end = initial_train_size

while train_end + PURGE_DAYS < n:

    test_start = train_end + PURGE_DAYS
    test_end = min(test_start + test_size, n)

    train = df.iloc[train_start:train_end].copy()

    test = df.iloc[test_start:test_end].copy()

    if len(test) == 0:
        break

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    logistic_model.fit(X_train, y_train)

    logistic_pred = logistic_model.predict(X_test)

    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    xgb_model = XGBClassifier(
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

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_pred
    )

    logistic_balanced_accuracy = balanced_accuracy_score(
        y_test,
        logistic_pred
    )

    logistic_auc = roc_auc_score(
        y_test,
        logistic_prob
    )


    xgb_accuracy = accuracy_score(
        y_test,
        xgb_pred
    )

    xgb_balanced_accuracy = balanced_accuracy_score(
        y_test,
        xgb_pred
    )

    xgb_auc = roc_auc_score(
        y_test,
        xgb_prob
    )


    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({
        "fold": fold,

        "train_start": train["date"].min(),
        "train_end": train["date"].max(),

        "test_start": test["date"].min(),
        "test_end": test["date"].max(),

        "train_rows": len(train),
        "test_rows": len(test),

        "logistic_accuracy": logistic_accuracy,
        "logistic_balanced_accuracy": logistic_balanced_accuracy,
        "logistic_auc": logistic_auc,

        "xgb_accuracy": xgb_accuracy,
        "xgb_balanced_accuracy": xgb_balanced_accuracy,
        "xgb_auc": xgb_auc
    })


    print(
        f"Fold {fold}: "
        f"Test {test['date'].min().date()} "
        f"to {test['date'].max().date()}"
    )

    print(
        f"  Logistic AUC: {logistic_auc:.4f} | "
        f"XGBoost AUC: {xgb_auc:.4f}"
    )

    print()


    # Move forward
    train_end = test_end

    fold += 1


# --------------------------------------------------
# Results DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 60)
print("WALK-FORWARD VALIDATION SUMMARY")
print("=" * 60)

print()

print("Number of folds:", len(results_df))

print()

print("Logistic Regression:")
print(
    "Mean Accuracy:",
    round(results_df["logistic_accuracy"].mean(), 4)
)

print(
    "Mean Balanced Accuracy:",
    round(
        results_df["logistic_balanced_accuracy"].mean(),
        4
    )
)

print(
    "Mean ROC-AUC:",
    round(results_df["logistic_auc"].mean(), 4)
)

print()

print("XGBoost:")
print(
    "Mean Accuracy:",
    round(results_df["xgb_accuracy"].mean(), 4)
)

print(
    "Mean Balanced Accuracy:",
    round(
        results_df["xgb_balanced_accuracy"].mean(),
        4
    )
)

print(
    "Mean ROC-AUC:",
    round(results_df["xgb_auc"].mean(), 4)
)

print()

print("Fold-by-fold results:")
print(
    results_df[
        [
            "fold",
            "logistic_accuracy",
            "logistic_balanced_accuracy",
            "logistic_auc",
            "xgb_accuracy",
            "xgb_balanced_accuracy",
            "xgb_auc"
        ]
    ]
)