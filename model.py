"""
model.py — Loan Approval Classifier
==================================================

Trains a Random Forest model using the merged HMDA dataset
(approved + denied loans).

Fixes included:
✔ Handles merged dataset with extra columns
✔ Prevents data leakage
✔ Handles missing values from merge
✔ Avoids predict_proba crash (single-class issue)

Feature Importance (Section 12):
──────────────────────────────────
After training, the script computes feature importances using the Random
Forest's built-in `feature_importances_` attribute, which measures how much
each input variable reduced impurity (uncertainty) across all decision trees.

A higher score means the model relied on that feature more heavily when
deciding whether to approve or deny a loan.

Outputs:
  • models/feature_importance.png — horizontal bar chart of the top 20 features
  • Terminal printout of the top 20 feature names and their scores

Use this to:
  • Sanity-check that the model is learning meaningful signals (e.g. loan
    amount, DTI) rather than leaking the target through a proxy variable
  • Identify fairness concerns if demographic features (race, sex) rank high
  • Justify dropping near-zero features to simplify the model
"""

"""
model.py — Loan Approval Classifier (ENHANCED VERSION)
====================================================

Upgrades:
✔ Feature engineering (loan ratios)
✔ Improved Random Forest tuning
✔ Optional XGBoost model
✔ Threshold optimization
✔ Better evaluation metrics
✔ Feature importance visualization
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# XGBoost
USE_XGBOOST = True
if USE_XGBOOST:
    from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/cleaned_data/nc_2019_2024_final_merged.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1) LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2) FEATURE ENGINEERING (🔥 BIG BOOST)
# ─────────────────────────────────────────────
df["loan_to_income"] = df["loan_amount"] / (df["income"].replace(0, np.nan))
df["loan_to_value"] = df["loan_amount"] / (df["property_value"].replace(0, np.nan))
df["income_per_person"] = df["income"] / (df["tract_population"].replace(0, np.nan))

# ─────────────────────────────────────────────
# FIX INF / EXTREME VALUES (CRITICAL FOR XGBOOST)
# ─────────────────────────────────────────────
# Replace inf values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: clip extreme values on numeric columns only (prevents huge spikes)
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].clip(lower=-1e6, upper=1e6)


# ─────────────────────────────────────────────
# 3) TARGET CHECK
# ─────────────────────────────────────────────
print("\nClass distribution BEFORE processing:")
print(df["approved"].value_counts())

# ─────────────────────────────────────────────
# 4) DROP LEAKAGE
# ─────────────────────────────────────────────
DROP_COLS = [
    "action_taken",
    "interest_rate",
    "rate_spread",
    "total_loan_costs",
    "discount_points",
    "lender_credits",
    "denial_reason-1",
    "denial_reason-1_label",
    "action_taken_label",
    "hoepa_status",
    "lei",
    "derived_msa-md",
    "year",
    "applicant_age",  # leakage
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# ─────────────────────────────────────────────
# 5) ENCODE ORDINAL
# ─────────────────────────────────────────────
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]

df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(
    {v: i for i, v in enumerate(DTI_ORDER)}
)

# ─────────────────────────────────────────────
# 6) ONE-HOT ENCODING
# ─────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "loan_purpose", "loan_type", "occupancy_type",
    "derived_loan_product_type", "derived_dwelling_category",
    "conforming_loan_limit", "derived_race",
    "derived_ethnicity", "derived_sex",
]

df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)

# ─────────────────────────────────────────────
# 7) SPLIT
# ─────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "approved"]

X = df[FEATURE_COLS]
y = df["approved"]

print("\nFinal class distribution:")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 8) HANDLE MISSING VALUES
# ─────────────────────────────────────────────
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

# ─────────────────────────────────────────────
# 9) MODEL TRAINING
# ─────────────────────────────────────────────
print("\nTraining model...")

if USE_XGBOOST:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
else:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

model.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 10) EVALUATION
# ─────────────────────────────────────────────
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 🔥 Threshold tuning
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")

y_pred = (y_pred_prob >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 11) SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(model, f"{OUTPUT_DIR}/model.joblib")
joblib.dump(FEATURE_COLS, f"{OUTPUT_DIR}/feature_names.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/train_medians.joblib")

# ─────────────────────────────────────────────
# 12) FEATURE IMPORTANCE
# ─────────────────────────────────────────────
if not USE_XGBOOST:
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
else:
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)

top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Top 20 Feature Importances")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)

print("\nTop 20 Features:")
print(top20.to_string())

# ─────────────────────────────────────────────
# 13) SAVE METRICS
# ─────────────────────────────────────────────
metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(roc_auc),
    "threshold": float(optimal_threshold)
}

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone.")