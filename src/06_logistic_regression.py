import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
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

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# --------------------------------------------------
# Purged time-based split
# --------------------------------------------------

split_index = int(len(df) * 0.60)

PURGE_DAYS = 5

train_end = split_index - PURGE_DAYS

train = df.iloc[:train_end].copy()
test = df.iloc[split_index:].copy()

# --------------------------------------------------
# Features and target
# --------------------------------------------------

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# --------------------------------------------------
# Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("Logistic Regression completed.")
print()

print("Train rows:", len(train))
print("Purged rows:", split_index - train_end)
print("Test rows:", len(test))
print()

print("Train period:")
print(train["date"].min().date(), "to", train["date"].max().date())
print()

print("Test period:")
print(test["date"].min().date(), "to", test["date"].max().date())
print()

print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))
print()

print("Confusion matrix:")
print(cm)
print()

print("Classification report:")
print(classification_report(y_test, y_pred))