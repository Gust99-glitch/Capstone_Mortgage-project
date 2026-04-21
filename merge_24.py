"""
merge_24.py — Combine Approved and Denied 24-Feature CSVs
==========================================================

Loads the approved and denied datasets, adds the target label,
drops leakage columns, and saves the combined file.

Output:
  • cleaned_data/nc_2019_2024_combined_24_features.csv
"""

import pandas as pd

APPROVED_PATH = "cleaned_data/nc_2019_2024_approved_24_features_filled.csv"
DENIED_PATH   = "cleaned_data/nc_2019_2024_denied_24_features_filled.csv"
OUTPUT_PATH   = "cleaned_data/nc_2019_2024_combined_24_features.csv"

# Load
approved = pd.read_csv(APPROVED_PATH)
denied   = pd.read_csv(DENIED_PATH)

print(f"Approved rows : {len(approved):,}")
print(f"Denied rows   : {len(denied):,}")

# Add target label
approved["approved"] = 1
denied["approved"]   = 0

# Combine
df = pd.concat([approved, denied], ignore_index=True)

# Drop leakage / metadata columns
df.drop(columns=["action_taken", "activity_year"], inplace=True, errors="ignore")

# Save
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nCombined rows : {len(df):,}")
print(f"Columns       : {df.shape[1]}")
print(f"Approval rate : {df['approved'].mean():.1%}")
print(f"\nSaved: {OUTPUT_PATH}")
