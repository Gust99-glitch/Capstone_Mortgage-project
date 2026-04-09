"""
lasso_selector.py — Automatic Feature Selection for Approval/Denial Dashboard
==============================================================================

Uses Logistic Regression with L1 penalty (Lasso equivalent for classification)
to automatically identify which HMDA fields drive loan approval vs denial.

No hand-picking — the model decides what matters.

Outputs:
  • models/lasso_selected_features.csv  — selected features + coefficients
  • models/lasso_model.joblib           — trained model
  • models/lasso_feature_importance.png — bar chart of top features
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/data_hmda_nc/nc_2019_2024.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1) LOAD & FILTER
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Full dataset: {len(df):,} rows")

# Keep only clear approvals (1) and denials (3)
df = df[df["action_taken"].isin([1, 3])].copy()
df["approved"] = (df["action_taken"] == 1).astype(int)
print(f"After filtering approved/denied: {len(df):,} rows")
print(f"Approval rate: {df['approved'].mean():.1%}")

# ─────────────────────────────────────────────
# 2) DROP LEAKAGE + IDENTIFIERS
# ─────────────────────────────────────────────
# These are post-decision fields — the model would never see them at application time
LEAKAGE = [
    "action_taken",
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "origination_charges",
    "total_points_and_fees",
    "discount_points",
    "lender_credits",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "hoepa_status",
    "purchaser_type",
    "initially_payable_to_institution",
    # AUS results are generated during underwriting (post-application)
    "aus-1", "aus-2", "aus-3", "aus-4", "aus-5",
]

# Identifiers / geo codes that are too granular or not useful for dashboard
IDENTIFIERS = [
    "lei", "census_tract", "state_code", "derived_msa-md", "activity_year", "year",
]

# Multi-value race/ethnicity sub-fields — use the cleaner derived_* versions instead
REDUNDANT = [
    "applicant_race-2", "applicant_race-3", "applicant_race-4", "applicant_race-5",
    "applicant_ethnicity-2", "applicant_ethnicity-3", "applicant_ethnicity-4", "applicant_ethnicity-5",
    "co-applicant_race-2", "co-applicant_race-3", "co-applicant_race-4", "co-applicant_race-5",
    "co-applicant_ethnicity-2", "co-applicant_ethnicity-3", "co-applicant_ethnicity-4", "co-applicant_ethnicity-5",
]

DROP_ALL = LEAKAGE + IDENTIFIERS + REDUNDANT
df.drop(columns=[c for c in DROP_ALL if c in df.columns], inplace=True)

# ─────────────────────────────────────────────
# 3) ENCODE DTI (ordinal)
# ─────────────────────────────────────────────
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]
df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(
    {v: i for i, v in enumerate(DTI_ORDER)}
)

# ─────────────────────────────────────────────
# 4) ONE-HOT ENCODE CATEGORICALS
# ─────────────────────────────────────────────
CATEGORICAL_COLS = [
    "loan_purpose", "loan_type", "occupancy_type",
    "derived_loan_product_type", "derived_dwelling_category",
    "conforming_loan_limit", "derived_race", "derived_ethnicity", "derived_sex",
    "applicant_age", "co-applicant_age",
    "applicant_credit_score_type", "co-applicant_credit_score_type",
    "applicant_race-1", "applicant_ethnicity-1",
    "co-applicant_race-1", "co-applicant_ethnicity-1",
    "applicant_sex", "co-applicant_sex",
    "construction_method", "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "preapproval", "business_or_commercial_purpose",
    "lien_status", "submission_of_application",
    "open-end_line_of_credit", "reverse_mortgage",
    "balloon_payment", "negative_amortization",
    "interest_only_payment", "other_nonamortizing_features",
]

# Only encode columns that actually exist in the dataframe
cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
df = pd.get_dummies(df, columns=cat_cols_present, dtype=int, drop_first=True)

# ─────────────────────────────────────────────
# 4b) COERCE REMAINING OBJECT COLUMNS TO NUMERIC
#     Some HMDA fields (e.g. loan_to_value_ratio) contain 'Exempt'
#     but aren't categorical — convert them, turning bad strings → NaN
# ─────────────────────────────────────────────
obj_cols = [c for c in df.select_dtypes(include="object").columns if c != "approved"]
if obj_cols:
    print(f"Coercing {len(obj_cols)} remaining string column(s) to numeric: {obj_cols}")
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ─────────────────────────────────────────────
# 5) SPLIT
# ─────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "approved"]
X = df[FEATURE_COLS]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 6) HANDLE MISSING VALUES
# ─────────────────────────────────────────────
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0)
X_test  = X_test.fillna(train_medians).fillna(0)

# ─────────────────────────────────────────────
# 7) SCALE (required for Lasso to work fairly)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 8) TRAIN LASSO (L1 Logistic Regression)
#    LogisticRegressionCV auto-picks best C via cross-validation
# ─────────────────────────────────────────────
print("\nTraining Lasso (L1 Logistic Regression) with cross-validation...")

model = LogisticRegressionCV(
    Cs=20,              # 20 candidate regularization strengths to try
    cv=5,               # 5-fold cross-validation
    penalty="l1",
    solver="liblinear",
    scoring="roc_auc",
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_sc, y_train)
print(f"Best C (regularization): {model.C_[0]:.4f}")

# ─────────────────────────────────────────────
# 9) EVALUATE
# ─────────────────────────────────────────────
y_prob = model.predict_proba(X_test_sc)[:, 1]
y_pred = model.predict(X_test_sc)

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print("\n--- RESULTS ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Denied", "Approved"]))

# ─────────────────────────────────────────────
# 10) EXTRACT SELECTED FEATURES
#     L1 zeros out unimportant features — survivors are what matter
# ─────────────────────────────────────────────
coefs = model.coef_[0]
feature_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": coefs,
})

# Keep only features the model kept (non-zero coefficient)
selected = feature_df[feature_df["coefficient"] != 0].copy()
selected["abs_coef"] = selected["coefficient"].abs()
selected = selected.sort_values("abs_coef", ascending=False).reset_index(drop=True)

# Direction label for dashboard use
selected["direction"] = selected["coefficient"].apply(
    lambda c: "increases approval odds" if c > 0 else "decreases approval odds"
)

print(f"\nFeatures selected by Lasso: {len(selected)} out of {len(FEATURE_COLS)} total")
print(f"Features zeroed out (not important): {len(FEATURE_COLS) - len(selected)}")

print("\nTop 30 most important features:")
print(selected[["feature", "coefficient", "direction"]].head(30).to_string(index=False))

# ─────────────────────────────────────────────
# 11) SAVE OUTPUTS
# ─────────────────────────────────────────────
selected.to_csv(f"{OUTPUT_DIR}/lasso_selected_features.csv", index=False)
joblib.dump(model,  f"{OUTPUT_DIR}/lasso_model.joblib")
joblib.dump(scaler, f"{OUTPUT_DIR}/lasso_scaler.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/lasso_medians.joblib")
joblib.dump(FEATURE_COLS, f"{OUTPUT_DIR}/lasso_feature_names.joblib")

print(f"\nSaved selected features → {OUTPUT_DIR}/lasso_selected_features.csv")

# ─────────────────────────────────────────────
# 12) PLOT
# ─────────────────────────────────────────────
top_n = selected.head(30).sort_values("coefficient")

colors = ["#d73027" if c < 0 else "#1a9850" for c in top_n["coefficient"]]

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(top_n["feature"], top_n["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Lasso Coefficient (L1 Logistic Regression)")
ax.set_title(
    "Top Features Driving Loan Approval / Denial\n"
    "(Green = increases approval odds | Red = decreases approval odds)"
)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/lasso_feature_importance.png", dpi=150)
print(f"Saved plot → {OUTPUT_DIR}/lasso_feature_importance.png")

print("\nDone.")
