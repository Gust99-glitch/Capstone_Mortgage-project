"""
xgboost_model.py — XGBoost Loan Approval Classifier (24 Features)
===================================================================

Trains an XGBoost model using the combined approved/denied dataset
with 24 Lasso-selected features.

Input:
  • cleaned_data/nc_2019_2024_combined_24_features.csv

Outputs:
  • models/xgb_model.joblib            — saved model
  • models/xgb_feature_names.joblib    — feature list used at training
  • models/xgb_train_medians.joblib    — training medians for imputation
  • models/xgb_feature_importance.png  — top 20 feature importance chart
  • models/xgb_metrics.json            — accuracy, ROC-AUC, threshold
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
DATA_PATH  = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/cleaned_data/nc_2019_2024_combined_24_features.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1) LOAD
# ─────────────────────────────────────────────
print("Loading combined 24-feature dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
print(f"Approval rate: {df['approved'].mean():.1%}")

# ─────────────────────────────────────────────
# 2) DTI ORDINAL ENCODING
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
# 3) ONE-HOT ENCODE CATEGORICALS
# ─────────────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "approved"]

if cat_cols:
    print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, dtype=np.int8, drop_first=True)

# ─────────────────────────────────────────────
# 4) CLEAN INF / EXTREME VALUES
# ─────────────────────────────────────────────
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ─────────────────────────────────────────────
# 5) SANITIZE COLUMN NAMES (XGBoost requirement)
# ─────────────────────────────────────────────
# XGBoost forbids [, ], and < in feature names (produced by age encoding).
df.columns = df.columns.str.replace(r"[<>\[\]]", "_", regex=True)

# ─────────────────────────────────────────────
# 6) SPLIT X / y
# ─────────────────────────────────────────────
X = df.drop(columns=["approved"])
y = df["approved"]

print(f"\nFeatures: {X.shape[1]}")
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
# 7) TRAIN XGBOOST
# ─────────────────────────────────────────────
print("\nTraining XGBoost...")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    verbosity=0,
)

xgb.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 8) EVALUATE
# ─────────────────────────────────────────────
y_pred_prob = xgb.predict_proba(X_test)[:, 1]

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
# 9) SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
feature_names = X.columns.tolist()

joblib.dump(xgb,                     f"{OUTPUT_DIR}/xgb_model.joblib")
joblib.dump(feature_names,           f"{OUTPUT_DIR}/xgb_feature_names.joblib")
joblib.dump(train_medians.to_dict(), f"{OUTPUT_DIR}/xgb_train_medians.joblib")

# ─────────────────────────────────────────────
# 10) FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importances = pd.Series(xgb.feature_importances_, index=feature_names)
top20       = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top20.sort_values().plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("Top 20 Feature Importances — XGBoost\n(NC HMDA 2019–2024, 24 Features)")
ax.set_xlabel("Feature Importance Score")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/xgb_feature_importance.png", dpi=150)
plt.close()

print("\nTop 20 Features:")
print(top20.to_string())

# ─────────────────────────────────────────────
# 11) SAVE METRICS
# ─────────────────────────────────────────────
metrics = {
    "model":           "XGBoost",
    "accuracy":        round(float(accuracy), 5),
    "roc_auc":         round(float(roc_auc), 5),
    "threshold":       round(float(optimal_threshold), 5),
    "n_features_used": len(feature_names),
}

with open(f"{OUTPUT_DIR}/xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
print("Done.")
