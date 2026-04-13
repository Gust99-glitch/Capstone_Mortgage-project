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

# lasso_selector.py — Stable + Smart Encoding Version

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from joblib import parallel_backend

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DATA_PATH  = r"C:\Users\jayso\.vscode\Capstone_Mortgage-project\data_hmda_nc\nc_2019_2024.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ROWS = 1_000_000   # reduce if needed

# ─────────────────────────────────────────────
# 1) LOAD
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

if MAX_ROWS and len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)
    print(f"Sampled down to {len(df):,} rows")

print(f"Dataset: {len(df):,} rows")

# Filter approved/denied
df = df[df["action_taken"].isin([1, 3])].copy()
df["approved"] = (df["action_taken"] == 1).astype(int)

print(f"After filter: {len(df):,}")
print(f"Approval rate: {df['approved'].mean():.1%}")

# ─────────────────────────────────────────────
# 2) DROP BAD COLUMNS
# ─────────────────────────────────────────────
DROP_COLS = [
    "action_taken","interest_rate","rate_spread","total_loan_costs",
    "origination_charges","total_points_and_fees","discount_points",
    "lender_credits","denial_reason-1","denial_reason-2",
    "denial_reason-3","denial_reason-4","hoepa_status",
    "purchaser_type","initially_payable_to_institution",
    "aus-1","aus-2","aus-3","aus-4","aus-5",
    "lei","census_tract","state_code","derived_msa-md",
    "activity_year","year"
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# ─────────────────────────────────────────────
# 3) DTI ENCODING
# ─────────────────────────────────────────────
DTI_ORDER = [
    "<20%","20%-<30%","30%-<36%","36%-<40%",
    "40%-<45%","45%-<50%","50%-60%",">60%","Exempt"
]

if "debt_to_income_ratio" in df.columns:
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(
        {v: i for i, v in enumerate(DTI_ORDER)}
    )

# ─────────────────────────────────────────────
# 4) SMART ENCODING (FIXED 🚨)
# ─────────────────────────────────────────────
print("\nSelecting categorical columns safely...")

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "approved"]

LOW_CARD_THRESHOLD = 20

low_card_cols = []
high_card_cols = []

for col in cat_cols:
    n_unique = df[col].nunique(dropna=True)

    if n_unique <= LOW_CARD_THRESHOLD:
        low_card_cols.append(col)
    else:
        high_card_cols.append(col)

print(f"Low-cardinality columns (encoded): {len(low_card_cols)}")
print(f"High-cardinality columns (reduced): {len(high_card_cols)}")

# 🔥 Reduce high-cardinality instead of dropping completely
TOP_N = 10

for col in high_card_cols:
    top_values = df[col].value_counts().nlargest(TOP_N).index
    df[col] = df[col].where(df[col].isin(top_values), "OTHER")

# Now safely encode everything
final_cat_cols = low_card_cols + high_card_cols

if final_cat_cols:
    df = pd.get_dummies(df, columns=final_cat_cols, dtype=np.int8, drop_first=True)

# ─────────────────────────────────────────────
# 5) SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=["approved"])
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 6) MISSING VALUES
# ─────────────────────────────────────────────
medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians).fillna(0)
X_test  = X_test.fillna(medians).fillna(0)

# ─────────────────────────────────────────────
# 7) SCALE
# ─────────────────────────────────────────────
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

# ─────────────────────────────────────────────
# 8) TRAIN MODEL
# ─────────────────────────────────────────────
print("\nTraining Lasso...")

model = LogisticRegressionCV(
    Cs=10,
    cv=3,
    penalty="l1",
    solver="saga",
    scoring="roc_auc",
    class_weight="balanced",
    max_iter=3000,
    random_state=42,
    n_jobs=-1
)

try:
    with parallel_backend("threading"):
        model.fit(X_train_sc, y_train)
except Exception:
    print("Fallback to single-core")
    model.n_jobs = 1
    model.fit(X_train_sc, y_train)

print(f"Best C: {model.C_[0]:.4f}")

# ─────────────────────────────────────────────
# 9) EVALUATE
# ─────────────────────────────────────────────
y_prob = model.predict_proba(X_test_sc)[:, 1]
y_pred = model.predict(X_test_sc)

print("\nRESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

# ─────────────────────────────────────────────
# 10) FEATURE IMPORTANCE
# ─────────────────────────────────────────────
coefs = model.coef_[0]

feature_df = pd.DataFrame({
    "feature": X.columns,
    "coef": coefs
})

selected = feature_df[feature_df["coef"] != 0].copy()
selected["abs"] = selected["coef"].abs()
selected = selected.sort_values("abs", ascending=False)

# ─────────────────────────────────────────────
# 11) SAVE
# ─────────────────────────────────────────────
selected.to_csv(f"{OUTPUT_DIR}/lasso_selected_features.csv", index=False)
joblib.dump(model, f"{OUTPUT_DIR}/lasso_model.joblib")

# ─────────────────────────────────────────────
# 12) PLOT
# ─────────────────────────────────────────────
top = selected.head(30).sort_values("coef")

plt.figure(figsize=(10,8))
plt.barh(top["feature"], top["coef"])
plt.axvline(0)
plt.title("Top Features (Lasso)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lasso_feature_importance.png")

print("\nDone.")