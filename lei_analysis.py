"""
lei_analysis.py — Lender Fairness Analysis Using LEI
=====================================================

Joins raw HMDA data with LEI institution names, computes approval
rates by race per lender, and ranks lenders by fairness.

Smaller gap = fairer lender = better recommendation for minority applicants.

Output:
  • lender_fairness.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data_hmda_nc/nc_2019_2024.csv"
LEI_PATH = "data_hmda_nc/US_lei_records.csv"
OUT_PATH = "lender_fairness.csv"
MIN_WHITE_BLACK = 100  # minimum loans for White and Black (largest groups)
MIN_ASIAN_NATIVE = 30  # minimum loans for Asian and Native (smaller groups)

RACES = [
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
]

# ─────────────────────────────────────────────
# 1) LOAD HMDA — LEI + RACE + ACTION TAKEN ONLY
# ─────────────────────────────────────────────
print("Loading HMDA data...")
df = pd.read_csv(
    RAW_PATH,
    usecols=lambda c: c in ["lei", "derived_race", "action_taken"],
    low_memory=False,
)
print(f"Loaded {len(df):,} rows")

df = df[df["action_taken"].isin([1, 3])].copy()
df["approved"] = (df["action_taken"] == 1).astype(int)
df = df[df["derived_race"].isin(RACES)]
print(f"After filtering: {len(df):,} rows")

# ─────────────────────────────────────────────
# 2) LOAD LEI INSTITUTION NAMES
# ─────────────────────────────────────────────
print("\nLoading LEI records...")
lei_df = pd.read_csv(
    LEI_PATH,
    usecols=["LEI", "Entity.LegalName", "Entity.LegalAddress.City", "Entity.LegalAddress.Region"],
    low_memory=False,
)
lei_df = lei_df.rename(columns={
    "LEI"                          : "lei",
    "Entity.LegalName"             : "institution_name",
    "Entity.LegalAddress.City"     : "city",
    "Entity.LegalAddress.Region"   : "state",
})
lei_df["lei"] = lei_df["lei"].str.strip().str.upper()
df["lei"]     = df["lei"].str.strip().str.upper()
print(f"Loaded {len(lei_df):,} LEI records")

# ─────────────────────────────────────────────
# 3) APPROVAL RATES BY LENDER + RACE
# ─────────────────────────────────────────────
print("\nComputing approval rates...")

grouped = (
    df.groupby(["lei", "derived_race"])["approved"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "approval_rate", "count": "loan_count"})
)

pivot  = grouped.pivot(index="lei", columns="derived_race", values="approval_rate").reset_index()
counts = grouped.pivot(index="lei", columns="derived_race", values="loan_count").reset_index()

pivot.columns.name  = None
counts.columns.name = None

pivot = pivot.rename(columns={
    "White"                           : "white_rate",
    "Black or African American"       : "black_rate",
    "Asian"                           : "asian_rate",
    "American Indian or Alaska Native" : "native_rate",
})
counts = counts.rename(columns={
    "White"                           : "white_count",
    "Black or African American"       : "black_count",
    "Asian"                           : "asian_count",
    "American Indian or Alaska Native" : "native_count",
})

lenders = pivot.merge(counts, on="lei")

# ─────────────────────────────────────────────
# 4) BIAS GAPS (vs White approval rate)
# ─────────────────────────────────────────────
lenders["black_gap"]  = (lenders["white_rate"] - lenders["black_rate"]).round(4)
lenders["asian_gap"]  = (lenders["white_rate"] - lenders["asian_rate"]).round(4)
lenders["native_gap"] = (lenders["white_rate"] - lenders["native_rate"]).round(4)

# Fairness score = Black gap only (most statistically reliable — largest sample size)
lenders["fairness_score"] = lenders["black_gap"].round(4)

# Total loans
total = df.groupby("lei")["approved"].count().reset_index().rename(columns={"approved": "total_loans"})
lenders = lenders.merge(total, on="lei")

# ─────────────────────────────────────────────
# 5) FILTER — minimum loan counts
# ─────────────────────────────────────────────
lenders = lenders[
    (lenders["white_count"] >= MIN_WHITE_BLACK) &
    (lenders["black_count"] >= MIN_WHITE_BLACK)
].copy()

# ─────────────────────────────────────────────
# 6) JOIN INSTITUTION NAMES
# ─────────────────────────────────────────────
lenders = lenders.merge(lei_df, on="lei", how="left")

# Round rates
for col in ["white_rate", "black_rate", "asian_rate", "native_rate"]:
    if col in lenders.columns:
        lenders[col] = lenders[col].round(4)

# Sort by fairness score
lenders = lenders.sort_values("fairness_score").reset_index(drop=True)
lenders["rank"] = lenders.index + 1

# ─────────────────────────────────────────────
# 7) SAVE
# ─────────────────────────────────────────────
cols = [
    "rank", "lei", "institution_name", "city", "state",
    "total_loans",
    "white_rate", "black_rate", "asian_rate", "native_rate",
    "black_gap", "asian_gap", "native_gap", "fairness_score",
    "white_count", "black_count", "asian_count", "native_count",
]
cols = [c for c in cols if c in lenders.columns]
lenders[cols].to_csv(OUT_PATH, index=False)

print(f"\nLenders analyzed : {len(lenders):,}")
print(f"\nTop 10 Fairest Lenders:")
print(lenders[["rank", "institution_name", "total_loans", "white_rate",
               "black_rate", "black_gap", "fairness_score"]].head(10).to_string(index=False))
print(f"\nSaved: {OUT_PATH}")
