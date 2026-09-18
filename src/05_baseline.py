import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

INPUT_PATH = "data/processed/webmelat_model_ready.csv"

# Load model-ready data
df = pd.read_csv(INPUT_PATH)

# Ensure chronological order
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

# Rows between train and test are intentionally discarded
purged_rows = df.iloc[train_end:split_index].copy()

# --------------------------------------------------
# Majority-class baseline
# --------------------------------------------------

majority_class = train["target"].mode()[0]

test["baseline_prediction"] = majority_class

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    test["target"],
    test["baseline_prediction"]
)

balanced_accuracy = balanced_accuracy_score(
    test["target"],
    test["baseline_prediction"]
)

cm = confusion_matrix(
    test["target"],
    test["baseline_prediction"]
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("Purged baseline model completed.")
print()
print("Train rows:", len(train))
print("Purged rows:", len(purged_rows))
print("Test rows:", len(test))
print()
print("Train period:")
print(train["date"].min().date(), "to", train["date"].max().date())
print()
print("Test period:")
print(test["date"].min().date(), "to", test["date"].max().date())
print()
print("Purged period:")
print(purged_rows["date"].min().date(), "to", purged_rows["date"].max().date())
print()
print("Majority class:", majority_class)
print("Test accuracy:", round(accuracy, 4))
print("Test balanced accuracy:", round(balanced_accuracy, 4))
print()
print("Confusion matrix:")
print(cm)