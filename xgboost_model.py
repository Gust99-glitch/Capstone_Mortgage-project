"""
xgboost_model.py — XGBoost Loan Approval Classifier
=====================================================

Trains an XGBoost model on the NC HMDA merged dataset (2019–2024)
using the 30 features selected by the Lasso logistic regression (lasso_selector.py).

Why XGBoost after Random Forest?
  Both are tree-based ensemble models, but XGBoost builds trees sequentially —
  each new tree corrects the errors of the previous one (boosting).
  This typically gives better accuracy than Random Forest on structured/tabular
  data like HMDA and is widely used in real-world credit/lending models.

Feature source:
  Same 30 features hardcoded from lasso_feature_importance.png, identical to
  rf_model.py, so results are directly comparable.

Outputs:
  • models/xgb_model.joblib            — saved XGBoost model
  • models/xgb_feature_names.joblib    — feature list used at training time
  • models/xgb_train_medians.joblib    — training medians for imputing missing values
  • models/xgb_feature_importance.png  — top 20 feature importance bar chart
  • models/xgb_metrics.json            — accuracy, ROC-AUC, optimal threshold
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
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
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/cleaned_data/nc_2019_2024_final_merged.csv"
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
# 1) LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

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
# XGBoost cannot handle inf values — replace with NaN for imputation later.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ─────────────────────────────────────────────
# 5) ONE-HOT ENCODE CATEGORICALS
# ─────────────────────────────────────────────
# Encode all remaining string columns so their values become binary columns.
# This produces named columns like derived_race_White, loan_term_Exempt, etc.
# drop_first=True avoids perfect multicollinearity (dummy variable trap).
cat_cols = [
    c for c in df.select_dtypes(include="object").columns
    if c != "approved"
]

if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, dtype=int, drop_first=True)

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
# 9) TRAIN XGBOOST
# ─────────────────────────────────────────────
# Key hyperparameters:
#   n_estimators=500    — 500 boosting rounds (trees built sequentially)
#   max_depth=6         — limits tree depth to prevent overfitting
#   learning_rate=0.05  — small steps so each tree makes a modest correction
#   subsample=0.8       — each tree sees 80% of rows (adds randomness, reduces overfit)
#   colsample_bytree=0.8 — each tree sees 80% of features (adds randomness)
#   scale_pos_weight    — compensates for class imbalance (denial/approval ratio)
print("\nTraining XGBoost...")

# Compute class weight to handle imbalance between approvals and denials
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0,
)

xgb.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 10) EVALUATE
# ─────────────────────────────────────────────
# Tune decision threshold using Youden's J (maximize TPR - FPR on ROC curve)
# rather than defaulting to 0.5, which can be suboptimal on imbalanced data.
y_pred_prob = xgb.predict_proba(X_test)[:, 1]

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
joblib.dump(xgb,                     f"{OUTPUT_DIR}/xgb_model.joblib")
joblib.dump(available,               f"{OUTPUT_DIR}/xgb_feature_names.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/xgb_train_medians.joblib")

# ─────────────────────────────────────────────
# 12) FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
# XGBoost importance = how many times a feature was used to split across all trees.
# Higher score = that feature was more frequently chosen as a decision point.
importances = pd.Series(xgb.feature_importances_, index=available)
top20       = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("Top 20 Feature Importances — XGBoost\n(NC HMDA 2019–2024, Lasso-selected features)")
ax.set_xlabel("Feature Importance Score")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/xgb_feature_importance.png", dpi=150)
plt.close()

print("\nTop 20 Features:")
print(top20.to_string())

# ─────────────────────────────────────────────
# 13) SAVE METRICS
# ─────────────────────────────────────────────
metrics = {
    "model":           "XGBoost",
    "accuracy":        float(accuracy),
    "roc_auc":         float(roc_auc),
    "threshold":       float(optimal_threshold),
    "n_features_used": len(available),
}

with open(f"{OUTPUT_DIR}/xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone.")
