"""
merge_datasets.py — Combine Approved + Denied HMDA Datasets (KEEP ALL COLUMNS)
=============================================================================

Purpose
-------
Merge datasets while preserving ALL columns (no dropping).

Key Idea
--------
- Columns that exist only in one dataset will be filled with NaN in the other
- This keeps full information without losing features

Output
------
cleaned_data/nc_2019_2024_final_merged.csv
"""

import pandas as pd

# ─────────────────────────────────────────────
# 1) LOAD DATASETS
# ─────────────────────────────────────────────
print("Loading datasets...")

df_main = pd.read_csv("/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/cleaned_data/nc_2019_2024_cleaned.csv",
    low_memory=False
)

df_denied = pd.read_csv(
    "/Users/gustave/Downloads/nc_2019_2024_denied_cleaned.csv",
    low_memory=False
)

print(f"Main dataset shape:   {df_main.shape}")
print(f"Denied dataset shape: {df_denied.shape}")

# ─────────────────────────────────────────────
# 2) MERGE DATASETS (KEEP ALL COLUMNS)
# ─────────────────────────────────────────────
# This automatically keeps ALL columns from both datasets
df = pd.concat([df_main, df_denied], ignore_index=True, sort=False)

print(f"Combined dataset shape: {df.shape}")
print(f"Total columns after merge: {len(df.columns)}")

# ─────────────────────────────────────────────
# 3) FILTER TARGET + CREATE LABEL
# ─────────────────────────────────────────────
APPROVED_CODES = {1, 2}
DENIED_CODES   = {3, 7}

df = df[df["action_taken"].isin(APPROVED_CODES | DENIED_CODES)].copy()

df["approved"] = df["action_taken"].apply(
    lambda x: 1 if x in APPROVED_CODES else 0
)

# ─────────────────────────────────────────────
# 4) SHUFFLE DATA
# ─────────────────────────────────────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# 5) CHECK DISTRIBUTION
# ─────────────────────────────────────────────
print("\nClass distribution:")
print(df["approved"].value_counts())
print(df["approved"].value_counts(normalize=True))

# ─────────────────────────────────────────────
# 6) CHECK MISSING VALUES (IMPORTANT NOW)
# ─────────────────────────────────────────────
print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ─────────────────────────────────────────────
# 7) SAVE FINAL DATASET
# ─────────────────────────────────────────────
OUTPUT_PATH = "cleaned_data/nc_2019_2024_final_merged.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved merged dataset to: {OUTPUT_PATH}")
print("Done.")