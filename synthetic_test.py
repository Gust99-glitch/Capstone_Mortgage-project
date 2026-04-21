"""
synthetic_test.py — Synthetic Applicant Profile Testing
========================================================

Creates realistic borrower profiles and runs them through both
the RF and XGBoost models to test predictions and fairness.

Scenarios:
  1. Fairness Test  — same financials, only race changes
  2. Risk Test      — same race, financial profile changes (DTI, solo vs joint)

Output:
  • synthetic_results.csv  — full results table
"""

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# LOAD MODELS & FEATURE NAMES
# ─────────────────────────────────────────────
rf         = joblib.load("models/rf_model.joblib")
rf_feats   = joblib.load("models/rf_feature_names.joblib")
rf_meds    = joblib.load("models/rf_train_medians.joblib")

xgb        = joblib.load("models/xgb_model.joblib")
xgb_feats  = joblib.load("models/xgb_feature_names.joblib")
xgb_meds   = joblib.load("models/xgb_train_medians.joblib")

RF_THRESHOLD  = 0.596
XGB_THRESHOLD = 0.500

# ─────────────────────────────────────────────
# DTI ENCODING MAP
# ─────────────────────────────────────────────
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]
DTI_MAP = {v: i for i, v in enumerate(DTI_ORDER)}

# ─────────────────────────────────────────────
# DEFINE SYNTHETIC PROFILES
# ─────────────────────────────────────────────
# Column order matches the 24 base features before encoding.
# Sentinel values used in HMDA data:
#   1111 = Exempt  |  9999 = No co-applicant  |  8888 = Not applicable

RACES = [
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
]

def base_profile():
    """Standard single-family home purchase, first lien, site-built."""
    return {
        "derived_dwelling_category"            : "Single Family (1-4 Units):Site-Built",
        "preapproval"                          : 2,       # not requested
        "loan_purpose"                         : 1,       # home purchase
        "lien_status"                          : 1,       # first lien
        "reverse_mortgage"                     : 2,       # not a reverse mortgage
        "open-end_line_of_credit"              : 2,       # not open-end
        "loan_amount"                          : 250000,
        "loan_to_value_ratio"                  : 80.0,
        "loan_term"                            : 360,     # 30-year
        "prepayment_penalty_term"              : -1,      # exempt
        "intro_rate_period"                    : -1,      # exempt
        "manufactured_home_secured_property_type"    : 1111,  # exempt
        "manufactured_home_land_property_interest"   : 1111,  # exempt
        "applicant_credit_score_type"          : 1,       # Equifax Beacon 5.0
        "co-applicant_ethnicity_observed"      : 3,       # not collected
        "co-applicant_sex"                     : 5,       # no co-applicant
        "co-applicant_sex_observed"            : 3,       # not applicable
        "ffiec_msa_md_median_family_income"    : 78000,
        "tract_to_msa_income_percentage"       : 105.0,
        "applicant_age"                        : "35-44",
        "co-applicant_age"                     : "9999",  # no co-applicant
    }

profiles = []

# ── SCENARIO 1: FAIRNESS TEST ─────────────────────────────────────────────────
# Three financial tiers, four races each — only race changes within each tier.

# Tier A: Low Risk (DTI <20%, 20% down, joint application)
for race in RACES:
    p = base_profile()
    p.update({
        "scenario"            : "Fairness — Low Risk",
        "derived_race"        : race,
        "debt_to_income_ratio": "<20%",
        "loan_to_value_ratio" : 80.0,
        "loan_amount"         : 200000,
        "co-applicant_age"    : "35-44",   # has co-applicant
        "co-applicant_sex"    : 2,
        "co-applicant_sex_observed": 2,
        "co-applicant_ethnicity_observed": 2,
    })
    profiles.append(p)

# Tier B: Medium Risk (DTI 36-40%, solo applicant)
for race in RACES:
    p = base_profile()
    p.update({
        "scenario"            : "Fairness — Medium Risk",
        "derived_race"        : race,
        "debt_to_income_ratio": "36%-<40%",
        "loan_to_value_ratio" : 90.0,
        "loan_amount"         : 300000,
    })
    profiles.append(p)

# Tier C: High Risk (DTI >60%, high LTV, solo applicant)
for race in RACES:
    p = base_profile()
    p.update({
        "scenario"            : "Fairness — High Risk",
        "derived_race"        : race,
        "debt_to_income_ratio": ">60%",
        "loan_to_value_ratio" : 97.0,
        "loan_amount"         : 400000,
    })
    profiles.append(p)

# ── SCENARIO 2: RISK TEST ─────────────────────────────────────────────────────
# White applicant, vary DTI and solo/joint to show how risk factors drive outcome.

risk_cases = [
    ("Risk — DTI <20%,  Joint", "<20%",      200000, "35-44", 2,  2, 2),
    ("Risk — DTI 30-36%, Joint", "30%-<36%", 250000, "35-44", 2,  2, 2),
    ("Risk — DTI 45-50%, Solo",  "45%-<50%", 300000, "9999",  5,  3, 3),
    ("Risk — DTI >60%,  Solo",   ">60%",     400000, "9999",  5,  3, 3),
]

for label, dti, amt, co_age, co_sex, co_sex_obs, co_eth in risk_cases:
    p = base_profile()
    p.update({
        "scenario"                       : label,
        "derived_race"                   : "White",
        "debt_to_income_ratio"           : dti,
        "loan_amount"                    : amt,
        "co-applicant_age"               : co_age,
        "co-applicant_sex"               : co_sex,
        "co-applicant_sex_observed"      : co_sex_obs,
        "co-applicant_ethnicity_observed": co_eth,
    })
    profiles.append(p)

# ─────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame(profiles)
scenarios  = df["scenario"]
races      = df["derived_race"]
df = df.drop(columns=["scenario"])

# ─────────────────────────────────────────────
# PREPROCESS — same steps as training
# ─────────────────────────────────────────────

# DTI ordinal encode
df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(DTI_MAP)

# One-hot encode categoricals
cat_cols = ["derived_dwelling_category", "derived_race", "applicant_age", "co-applicant_age"]
df = pd.get_dummies(df, columns=cat_cols, dtype=np.int8)

# ─────────────────────────────────────────────
# ALIGN TO MODEL FEATURE SETS
# ─────────────────────────────────────────────
def align(frame, feature_names, medians):
    """Reindex to match training columns, fill missing with 0."""
    X = frame.reindex(columns=feature_names, fill_value=0)
    X = X.fillna(medians).fillna(0)
    return X

# RF
X_rf  = align(df, rf_feats, rf_meds)

# XGB — sanitize column names
xgb_feats_clean = [
    f.replace("<", "_").replace(">", "_").replace("[", "_").replace("]", "_")
    for f in xgb_feats
]
df_xgb = df.copy()
df_xgb.columns = df_xgb.columns.str.replace(r"[<>\[\]]", "_", regex=True)
X_xgb = align(df_xgb, xgb_feats_clean, xgb_meds)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
rf_probs  = rf.predict_proba(X_rf)[:, 1]
rf_preds  = (rf_probs  >= RF_THRESHOLD).astype(int)

xgb_probs = xgb.predict_proba(X_xgb)[:, 1]
xgb_preds = (xgb_probs >= XGB_THRESHOLD).astype(int)

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame({
    "Scenario"          : scenarios.values,
    "Race"              : races.values,
    "RF_Prob_%"         : (rf_probs  * 100).round(1),
    "RF_Decision"       : ["APPROVED" if p == 1 else "DENIED" for p in rf_preds],
    "XGB_Prob_%"        : (xgb_probs * 100).round(1),
    "XGB_Decision"      : ["APPROVED" if p == 1 else "DENIED" for p in xgb_preds],
})

# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 35)

print("\n" + "=" * 100)
print("  SYNTHETIC APPLICANT TEST RESULTS")
print("=" * 100)

for scenario in results["Scenario"].unique():
    subset = results[results["Scenario"] == scenario]
    print(f"\n{'─'*100}")
    print(f"  {scenario}")
    print(f"{'─'*100}")
    print(subset[["Race", "RF_Prob_%", "RF_Decision", "XGB_Prob_%", "XGB_Decision"]].to_string(index=False))

print("\n" + "=" * 100)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
results.to_csv("synthetic_results.csv", index=False)
print("\nSaved: synthetic_results.csv")
