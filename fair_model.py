"""
fair_model.py — Fair Loan Approval Model (Financial Features Only)
===================================================================

Trains a Random Forest model using ONLY financial features — no race,
no demographic information. The goal is to predict loan approval based
purely on financial risk, with no racial signal.

When compared to the main model (rf_model.py), the gap between the two
predictions for the same applicant reveals the racial penalty embedded
in historical lending data.

  Main model prediction  →  what history says (includes racial signal)
  Fair model prediction  →  what finances alone say (no racial signal)
  Gap                    →  racial penalty / bias gap

DEMOGRAPHIC COLUMNS REMOVED:
  - derived_race
  - applicant_age
  - co-applicant_age
  - co-applicant_sex
  - co-applicant_sex_observed
  - co-applicant_ethnicity_observed

FINANCIAL FEATURES KEPT (17):
  derived_dwelling_category, preapproval, loan_purpose, lien_status,
  reverse_mortgage, open-end_line_of_credit, loan_amount,
  loan_to_value_ratio, loan_term, prepayment_penalty_term,
  intro_rate_period, manufactured_home_secured_property_type,
  manufactured_home_land_property_interest, debt_to_income_ratio,
  applicant_credit_score_type, ffiec_msa_md_median_family_income,
  tract_to_msa_income_percentage

Note: ffiec_msa_md_median_family_income and tract_to_msa_income_percentage
are area-level financial indicators. They may carry indirect racial signal
due to historical redlining (wealthier areas tend to be less diverse), but
they reflect legitimate financial geography rather than individual demographics.

Input:
  • cleaned_data/nc_2019_2024_combined_24_features.csv

Outputs:
  • models/fair_model.joblib            — saved Fair RF model
  • models/fair_feature_names.joblib    — feature list used at training
  • models/fair_train_medians.joblib    — training medians for imputation
  • models/fair_feature_importance.png  — top feature importance chart
  • models/fair_metrics.json            — accuracy, ROC-AUC, threshold
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/cleaned_data/nc_2019_2024_combined_24_features.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DEMOGRAPHIC COLUMNS TO REMOVE
# ─────────────────────────────────────────────
DEMOGRAPHIC_COLS = [
    "derived_race",
    "applicant_age",
    "co-applicant_age",
    "co-applicant_sex",
    "co-applicant_sex_observed",
    "co-applicant_ethnicity_observed",
]

# ─────────────────────────────────────────────
# 1) LOAD
# ─────────────────────────────────────────────
print("Loading combined 24-feature dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
print(f"Approval rate: {df['approved'].mean():.1%}")

# ─────────────────────────────────────────────
# 2) DROP DEMOGRAPHIC COLUMNS
# ─────────────────────────────────────────────
dropped = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"\nDropped demographic columns: {dropped}")
print(f"Remaining columns: {df.shape[1]} (including target)")

# ─────────────────────────────────────────────
# 3) DTI ORDINAL ENCODING
# ─────────────────────────────────────────────
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]

if "debt_to_income_ratio" in df.columns:
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(
        {v: i for i, v in enumerate(DTI_ORDER)}
    )

# ─────────────────────────────────────────────
# 4) ONE-HOT ENCODE CATEGORICALS
# ─────────────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "approved"]

if cat_cols:
    print(f"\nEncoding {len(cat_cols)} categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, dtype=np.int8, drop_first=True)

# ─────────────────────────────────────────────
# 5) CLEAN INF / EXTREME VALUES
# ─────────────────────────────────────────────
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ─────────────────────────────────────────────
# 6) SPLIT X / y
# ─────────────────────────────────────────────
X = df.drop(columns=["approved"])
y = df["approved"]

print(f"\nFinancial features used: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts(normalize=True).round(3)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 7) IMPUTE MISSING VALUES
# ─────────────────────────────────────────────
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0)
X_test  = X_test.fillna(train_medians).fillna(0)

# ─────────────────────────────────────────────
# 8) TRAIN FAIR RANDOM FOREST
# ─────────────────────────────────────────────
print("\nTraining Fair Random Forest (no demographic features)...")

fair_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

fair_rf.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 9) EVALUATE
# ─────────────────────────────────────────────
y_pred_prob = fair_rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
optimal_idx       = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

y_pred   = (y_pred_prob >= optimal_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print(f"\nOptimal threshold : {optimal_threshold:.3f}")
print(f"Accuracy          : {accuracy:.4f}")
print(f"ROC-AUC           : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 10) SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
feature_names = X.columns.tolist()

joblib.dump(fair_rf,                 f"{OUTPUT_DIR}/fair_model.joblib")
joblib.dump(feature_names,           f"{OUTPUT_DIR}/fair_feature_names.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/fair_train_medians.joblib")

# ─────────────────────────────────────────────
# 11) FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importances = pd.Series(fair_rf.feature_importances_, index=feature_names)
top20       = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax, color="seagreen")
ax.set_title("Feature Importances — Fair Model (Financial Only)\n(NC HMDA 2019–2024, No Demographic Features)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fair_feature_importance.png", dpi=150)
plt.close()

print("\nTop 20 Features:")
print(top20.to_string())

# ─────────────────────────────────────────────
# 12) SAVE METRICS
# ─────────────────────────────────────────────
metrics = {
    "model"            : "FairRandomForest",
    "demographic_cols_removed": dropped,
    "accuracy"         : round(float(accuracy), 5),
    "roc_auc"          : round(float(roc_auc), 5),
    "threshold"        : round(float(optimal_threshold), 5),
    "n_features_used"  : len(feature_names),
}

with open(f"{OUTPUT_DIR}/fair_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
print("\nDone.")
