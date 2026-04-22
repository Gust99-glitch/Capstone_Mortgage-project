"""
fair_synthetic_test.py — Test Fair Model on Synthetic Profiles
==============================================================
Same scenarios as synthetic_test.py but run through the fair model
(financial features only — no race, no demographics).
"""

import joblib
import numpy as np
import pandas as pd

BASE = "/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project"

fair       = joblib.load(f"{BASE}/models/fair_model.joblib")
fair_feats = joblib.load(f"{BASE}/models/fair_feature_names.joblib")
fair_meds  = joblib.load(f"{BASE}/models/fair_train_medians.joblib")

FAIR_THRESHOLD = 0.570

DTI_MAP = {v: i for i, v in enumerate([
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
])}

# ─────────────────────────────────────────────
# PROFILES (no demographic columns)
# ─────────────────────────────────────────────
def base():
    return {
        "derived_dwelling_category"                : "Single Family (1-4 Units):Site-Built",
        "preapproval"                              : 2,
        "loan_purpose"                             : 1,
        "lien_status"                              : 1,
        "reverse_mortgage"                         : 2,
        "open-end_line_of_credit"                  : 2,
        "loan_amount"                              : 250000,
        "loan_to_value_ratio"                      : 80.0,
        "loan_term"                                : 360,
        "prepayment_penalty_term"                  : -1,
        "intro_rate_period"                        : -1,
        "manufactured_home_secured_property_type"  : 1111,
        "manufactured_home_land_property_interest" : 1111,
        "applicant_credit_score_type"              : 1,
        "ffiec_msa_md_median_family_income"        : 78000,
        "tract_to_msa_income_percentage"           : 105.0,
    }

profiles = [
    {**base(), "scenario": "Low Risk  (DTI <20%, LTV 80%, Joint)",    "debt_to_income_ratio": "<20%",      "loan_amount": 200000, "loan_to_value_ratio": 80.0},
    {**base(), "scenario": "Medium Risk (DTI 36-40%, LTV 90%, Solo)", "debt_to_income_ratio": "36%-<40%",  "loan_amount": 300000, "loan_to_value_ratio": 90.0},
    {**base(), "scenario": "High Risk  (DTI >60%, LTV 97%, Solo)",    "debt_to_income_ratio": ">60%",      "loan_amount": 400000, "loan_to_value_ratio": 97.0},
    {**base(), "scenario": "Risk — DTI <20%,  Joint",                  "debt_to_income_ratio": "<20%",      "loan_amount": 200000},
    {**base(), "scenario": "Risk — DTI 30-36%, Joint",                 "debt_to_income_ratio": "30%-<36%",  "loan_amount": 250000},
    {**base(), "scenario": "Risk — DTI 45-50%, Solo",                  "debt_to_income_ratio": "45%-<50%",  "loan_amount": 300000},
    {**base(), "scenario": "Risk — DTI >60%,  Solo",                   "debt_to_income_ratio": ">60%",      "loan_amount": 400000},
]

df = pd.DataFrame(profiles)
scenarios = df.pop("scenario")

# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(DTI_MAP)
df = pd.get_dummies(df, columns=["derived_dwelling_category"], dtype=np.int8)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

X = df.reindex(columns=fair_feats, fill_value=0).fillna(fair_meds).fillna(0)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
probs = fair.predict_proba(X)[:, 1]
preds = ["APPROVED" if p >= FAIR_THRESHOLD else "DENIED" for p in probs]

results = pd.DataFrame({
    "Scenario"       : scenarios.values,
    "Fair_Model_%"   : (probs * 100).round(1),
    "Decision"       : preds,
})

print("\n" + "=" * 70)
print("  FAIR MODEL — SYNTHETIC TEST (Financial Features Only)")
print("=" * 70)
print(results.to_string(index=False))
print("=" * 70)

results.to_csv("fair_synthetic_results.csv", index=False)
print("\nSaved: fair_synthetic_results.csv")
