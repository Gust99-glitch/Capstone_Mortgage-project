"""
rf_model.py — Random Forest Loan Approval Classifier
=====================================================

Trains a Random Forest model on the NC HMDA raw cleaned dataset (2019–2024)
using the 30 features selected by the Lasso logistic regression (lasso_selector.py).

Why Random Forest?
  Lasso is linear it can only learn straight-line relationships.
  Random Forest builds hundreds of decision trees and combines them,
  capturing non-linear patterns (e.g., DTI risk jumps above 50%, not linearly).
  This typically produces meaningfully better accuracy on HMDA tabular data.

Feature source:
  Features are hardcoded from lasso_feature_importance.png — the Lasso model's
  non-zero coefficients, which represent the features it found most predictive
  of loan approval vs denial.

Outputs:
  • models/rf_model.joblib            — saved Random Forest model
  • models/rf_feature_names.joblib    — feature list used at training time
  • models/rf_train_medians.joblib    — training medians for imputing missing values
  • models/rf_feature_importance.png  — top 20 feature importance bar chart
  • models/rf_metrics.json            — accuracy, ROC-AUC, optimal threshold
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
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/data_hmda_nc/nc_2019_2024.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LASSO-SELECTED FEATURES (hardcoded from lasso_feature_importance.png)
# ─────────────────────────────────────────────
# These are the 30 features that Lasso assigned non-zero coefficients to,
# meaning the model found them genuinely predictive of approval/denial.
# Encoded column names (e.g. derived_race_White) are produced in Section 5 below.
LASSO_FEATURES = [
    "applicant_credit_score_type",
    "preapproval",
    "debt_to_income_ratio",
    "derived_dwelling_category_Single Family (1-4 Units):Manufactured",
    "loan_to_value_ratio_OTHER",
    "co-applicant_age_9999",
    "co-applicant_ethnicity_observed",
    "manufactured_home_land_property_interest",
    "loan_purpose",
    "lien_status",
    "loan_term_276",
    "prepayment_penalty_term_36",
    "applicant_age_65-74",
    "applicant_age_55-64",
    "applicant_age_>74",
    "co-applicant_sex_observed",
    "intro_rate_period_60",
    "ffiec_msa_md_median_family_income",
    "derived_race_Asian",
    "intro_rate_period_OTHER",
    "tract_to_msa_income_percentage",
    "derived_race_Race Not Available",
    "intro_rate_period_Exempt",
    "manufactured_home_secured_property_type",
    "co-applicant_sex",
    "derived_race_White",
    "loan_amount",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "loan_term_Exempt",
]

# ─────────────────────────────────────────────
# 1) LOAD ONLY NEEDED COLUMNS
# ─────────────────────────────────────────────
# LASSO_FEATURES contains post-encoding names (e.g. derived_race_White),
# but the raw CSV has the base column names (e.g. derived_race).
# We load the base columns here, encode them in Section 5, then select
# the Lasso features by their encoded names.
BASE_COLS = [
    "action_taken",
    "applicant_credit_score_type",
    "preapproval",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_purpose",
    "lien_status",
    "loan_term",
    "loan_to_value_ratio",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "prepayment_penalty_term",
    "intro_rate_period",
    "derived_race",
    "derived_dwelling_category",
    "applicant_age",
    "co-applicant_age",
    "co-applicant_sex",
    "co-applicant_sex_observed",
    "co-applicant_ethnicity_observed",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
]

print("Loading data (selected columns only)...")
df = pd.read_csv(DATA_PATH, usecols=lambda c: c in BASE_COLS, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 1b) FILTER TO APPROVED / DENIED ONLY
# ─────────────────────────────────────────────
# action_taken == 1 → loan originated (approved)
# action_taken == 3 → application denied
# All other values (withdrawn, incomplete, etc.) are excluded because they
# don't represent a clear approval/denial decision.
df = df[df["action_taken"].isin([1, 3])].copy()
df["approved"] = (df["action_taken"] == 1).astype(int)
print(f"After filtering approved/denied: {len(df):,} rows")
print(f"Approval rate: {df['approved'].mean():.1%}")

# ─────────────────────────────────────────────
# 1c) SAMPLE TO PREVENT MEMORY CRASH
# ─────────────────────────────────────────────
# 500k rows is enough for stable, representative model training.
MAX_ROWS = 500_000
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)
    print(f"Sampled down to {len(df):,} rows")

# ─────────────────────────────────────────────
# 2) DROP LEAKAGE COLUMNS
# ─────────────────────────────────────────────
# Remove columns that directly encode the outcome or are only known
# after a decision is made — using them would inflate performance.
DROP_COLS = [
    "action_taken", "action_taken_label",
    "interest_rate", "rate_spread", "total_loan_costs",
    "origination_charges", "total_points_and_fees",
    "discount_points", "lender_credits",
    "denial_reason-1", "denial_reason-1_label",
    "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "hoepa_status", "purchaser_type",
    "initially_payable_to_institution",
    "aus-1", "aus-2", "aus-3", "aus-4", "aus-5",
    "lei", "census_tract", "state_code",
    "derived_msa-md", "activity_year", "year",
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# ─────────────────────────────────────────────
# 3) DTI ORDINAL ENCODING
# ─────────────────────────────────────────────
# debt_to_income_ratio is stored as a string range (e.g. "30%-<36%").
# Map it to an ordered integer so the model treats it as a numeric risk level.
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]

if "debt_to_income_ratio" in df.columns:
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(
        {v: i for i, v in enumerate(DTI_ORDER)}
    )

# ─────────────────────────────────────────────
# 4) INF / EXTREME VALUE CLEANUP
# ─────────────────────────────────────────────
# Replace infinite values with NaN so they get imputed cleanly later.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ─────────────────────────────────────────────
# 5) ONE-HOT ENCODE CATEGORICALS (targeted — only needed dummies)
# ─────────────────────────────────────────────
# Instead of get_dummies on all object columns (which creates a huge wide
# DataFrame before we can drop anything), we manually create only the specific
# dummy columns that appear in LASSO_FEATURES. This avoids the memory spike
# that was killing the process on machines with limited RAM.

def _make_targeted_dummies(frame, lasso_features):
    """Create only the dummy columns referenced in lasso_features."""
    object_cols = set(frame.select_dtypes(include="object").columns) - {"approved"}

    for feat in lasso_features:
        for col in object_cols:
            prefix = col + "_"
            if feat.startswith(prefix):
                value = feat[len(prefix):]
                frame[feat] = (frame[col] == value).astype(np.int8)
                break

    frame.drop(columns=list(object_cols), inplace=True)
    return frame

df = _make_targeted_dummies(df, LASSO_FEATURES)

# ─────────────────────────────────────────────
# 6) SELECT LASSO FEATURES
# ─────────────────────────────────────────────
# Keep only the 30 features Lasso selected. Some may be missing if encoding
# produced slightly different column names — we warn rather than crash.
available = [f for f in LASSO_FEATURES if f in df.columns]
missing   = [f for f in LASSO_FEATURES if f not in df.columns]

if missing:
    print(f"\n[WARNING] {len(missing)} Lasso features not found after encoding:")
    for f in missing:
        print(f"  - {f}")

print(f"\nUsing {len(available)}/{len(LASSO_FEATURES)} Lasso features.")

X = df[available]
y = df["approved"]

print("\nClass distribution:")
print(y.value_counts(normalize=True).round(3))

# ─────────────────────────────────────────────
# 7) TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
# stratify=y keeps the approval/denial ratio equal in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 8) IMPUTE MISSING VALUES
# ─────────────────────────────────────────────
# Use training set medians only — never fit on test data.
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0)
X_test  = X_test.fillna(train_medians).fillna(0)

# ─────────────────────────────────────────────
# 9) TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
# Key hyperparameters:
#   n_estimators=300    — 300 trees for stable, averaged predictions
#   max_depth=None      — trees grow fully; controls via min_samples_leaf
#   min_samples_leaf=5  — each leaf needs at least 5 samples (prevents overfitting)
#   max_features="sqrt" — each split considers sqrt(n_features) candidates (standard)
#   class_weight="balanced" — adjusts for class imbalance between approvals/denials
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 10) EVALUATE
# ─────────────────────────────────────────────
# Tune decision threshold using Youden's J (maximize TPR - FPR on ROC curve)
# rather than defaulting to 0.5, which can be suboptimal on imbalanced data.
y_pred_prob = rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
optimal_idx       = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")

y_pred   = (y_pred_prob >= optimal_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print("\n--- RESULTS ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 11) SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(rf,                      f"{OUTPUT_DIR}/rf_model.joblib")
joblib.dump(available,               f"{OUTPUT_DIR}/rf_feature_names.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/rf_train_medians.joblib")

# ─────────────────────────────────────────────
# 12) FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
# Random Forest importance = mean decrease in Gini impurity across all trees.
# Higher score = that feature was more useful for splitting decisions.
importances = pd.Series(rf.feature_importances_, index=available)
top20       = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances — Random Forest\n(NC HMDA 2019–2024, Lasso-selected features)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/rf_feature_importance.png", dpi=150)
plt.close()

print("\nTop 20 Features:")
print(top20.to_string())

# ─────────────────────────────────────────────
# 13) SAVE METRICS
# ─────────────────────────────────────────────
metrics = {
    "model":            "RandomForest",
    "accuracy":         float(accuracy),
    "roc_auc":          float(roc_auc),
    "threshold":        float(optimal_threshold),
    "n_features_used":  len(available),
}

with open(f"{OUTPUT_DIR}/rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone.")
