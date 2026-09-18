import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

FILES = {
    "stage_20_14": (
        BASE / "data" / "processed" /
        "integrity_sensitivity_results.csv"
    ),

    "stage_20_15": (
        BASE / "data" / "processed" /
        "date_aligned_sensitivity_results.csv"
    ),

    "stage_20_16": (
        BASE / "data" / "processed" /
        "common_test_period_sensitivity_results.csv"
    ),

    "stage_20_17": (
        BASE / "data" / "processed" /
        "training_size_controlled_results.csv"
    ),
}

OUTPUT = (
    BASE / "data" / "processed" /
    "final_integrity_conclusion.csv"
)


print()
print("=" * 70)
print("STAGE 20.18: FINAL INTEGRITY CONCLUSION")
print("=" * 70)
print()


# =========================================================
# 1. CHECK FILES
# =========================================================

print("Checking Stage 20 files...")
print()

for name, path in FILES.items():

    if path.exists():

        print(
            f"[OK] {name}: {path.name}"
        )

    else:

        print(
            f"[MISSING] {name}: {path}"
        )


# =========================================================
# 2. LOAD RESULTS
# =========================================================

results = []


# ---------------------------------------------------------
# Stage 20.14
# ---------------------------------------------------------

if FILES["stage_20_14"].exists():

    df = pd.read_csv(
        FILES["stage_20_14"]
    )

    for _, row in df.iterrows():

        results.append({
            "stage": "20.14",
            "analysis": "Integrity sensitivity",
            "dataset": row.get("dataset"),
            "train_rows": row.get("train_rows"),
            "test_rows": row.get("test_rows"),
            "accuracy": row.get("accuracy"),
            "balanced_accuracy": row.get(
                "balanced_accuracy"
            ),
            "roc_auc": row.get("roc_auc"),
            "conclusion":
                "Sensitivity to event-aware cleaning "
                "and structural-break exclusion was tested."
        })


# ---------------------------------------------------------
# Stage 20.15
# ---------------------------------------------------------

if FILES["stage_20_15"].exists():

    df = pd.read_csv(
        FILES["stage_20_15"]
    )

    for _, row in df.iterrows():

        results.append({
            "stage": "20.15",
            "analysis": "Date-aligned sensitivity",
            "dataset": row.get("dataset"),
            "train_rows": row.get("train_rows"),
            "test_rows": row.get("test_rows"),
            "accuracy": row.get("accuracy"),
            "balanced_accuracy": row.get(
                "balanced_accuracy"
            ),
            "roc_auc": row.get("roc_auc"),
            "conclusion":
                "A fixed temporal test boundary was used."
        })


# ---------------------------------------------------------
# Stage 20.16
# ---------------------------------------------------------

if FILES["stage_20_16"].exists():

    df = pd.read_csv(
        FILES["stage_20_16"]
    )

    for _, row in df.iterrows():

        results.append({
            "stage": "20.16",
            "analysis": "Common test period",
            "dataset": row.get("dataset"),
            "train_rows": row.get("train_rows"),
            "test_rows": row.get("test_rows"),
            "accuracy": row.get("accuracy"),
            "balanced_accuracy": row.get(
                "balanced_accuracy"
            ),
            "roc_auc": row.get("roc_auc"),
            "conclusion":
                "All scenarios were evaluated on "
                "the same common test period."
        })


# ---------------------------------------------------------
# Stage 20.17
# ---------------------------------------------------------

if FILES["stage_20_17"].exists():

    df = pd.read_csv(
        FILES["stage_20_17"]
    )

    for _, row in df.iterrows():

        results.append({
            "stage": "20.17",
            "analysis": "Training-size controlled",
            "dataset": row.get("dataset"),
            "train_rows": row.get("train_rows"),
            "test_rows": row.get("test_rows"),
            "accuracy": row.get("accuracy"),
            "balanced_accuracy": row.get(
                "balanced_accuracy"
            ),
            "roc_auc": row.get("roc_auc"),
            "conclusion":
                "Training size was controlled at "
                "the Clean dataset size."
        })


# =========================================================
# 3. CREATE SUMMARY
# =========================================================

summary = pd.DataFrame(
    results
)


# =========================================================
# 4. FINAL DATASET DECISION
# =========================================================

decision = pd.DataFrame([
    {
        "decision_area":
            "Primary modeling dataset",

        "decision":
            "ORIGINAL",

        "reason":
            "Event-aware cleaning reduced predictive "
            "performance in common-test and "
            "training-size-controlled analyses."
    },

    {
        "decision_area":
            "Structural-break exclusion",

        "decision":
            "NOT REQUIRED",

        "reason":
            "Removing the identified structural-break "
            "window produced essentially no change "
            "in model performance."
    },

    {
        "decision_area":
            "Event flags",

        "decision":
            "RETAIN",

        "reason":
            "Event flags remain valuable for audit, "
            "risk analysis and future robustness checks."
    },

    {
        "decision_area":
            "Clean dataset",

        "decision":
            "RETAIN AS SENSITIVITY DATASET",

        "reason":
            "It should not be discarded, but it is "
            "not selected as the primary modeling dataset."
    },

    {
        "decision_area":
            "Original raw data",

        "decision":
            "RETAIN",

        "reason":
            "Raw TSETMC data remains the immutable "
            "source layer."
    }
])


# =========================================================
# 5. PRINT RESULTS
# =========================================================

print()
print("=" * 70)
print("INTEGRITY ANALYSIS SUMMARY")
print("=" * 70)
print()

if len(summary) > 0:

    display_columns = [
        "stage",
        "analysis",
        "dataset",
        "train_rows",
        "test_rows",
        "accuracy",
        "balanced_accuracy",
        "roc_auc"
    ]

    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

else:

    print(
        "No Stage 20 results were found."
    )


print()
print("=" * 70)
print("FINAL DATASET DECISION")
print("=" * 70)
print()

print(
    decision.to_string(index=False)
)


# =========================================================
# 6. SAVE
# =========================================================

summary_file = (
    BASE / "data" / "processed" /
    "final_integrity_analysis_summary.csv"
)

decision_file = (
    BASE / "data" / "processed" /
    "final_dataset_decision.csv"
)

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

decision.to_csv(
    decision_file,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 7. FINAL MESSAGE
# =========================================================

print()
print("=" * 70)
print("STAGE 20.18 COMPLETED")
print("=" * 70)
print()

print(
    "Primary dataset for next stages: ORIGINAL"
)

print(
    "Clean dataset: retained for sensitivity analysis"
)

print(
    "Structural-break exclusion: not required"
)

print(
    "Event flags: retained"
)

print()
print(
    f"Saved summary: {summary_file}"
)

print(
    f"Saved decision: {decision_file}"
)

print()
print(
    "NEXT: STAGE 21 - FINAL DATASET & MODEL VALIDATION"
)