import pandas as pd
import os

# -----------------------------------------
# FILE PATHS
# -----------------------------------------
hmda_file = r"C:\Users\Carlin Crawford\Desktop\DTSC 4301\Capstone_Mortgage-project\data_hmda_nc\nc_2019_2024.csv"
lei_lookup_file = r"C:\Users\Carlin Crawford\Downloads\updated_lei_records (1).csv"
out_dir = r"C:\Users\Carlin Crawford\Desktop\DTSC 4301\Capstone_Mortgage-project\cleaned_data"

# -----------------------------------------
# SETTINGS
# -----------------------------------------
min_apps = 100   # change to 50, 100, or 250 if you want

# -----------------------------------------
# LOAD FILES
# -----------------------------------------
hmda = pd.read_csv(hmda_file, low_memory=False)
lei_lookup = pd.read_csv(lei_lookup_file, low_memory=False)

# -----------------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------------
hmda.columns = hmda.columns.str.strip()
lei_lookup.columns = lei_lookup.columns.str.strip()

# -----------------------------------------
# CHECK REQUIRED COLUMNS
# -----------------------------------------
required_hmda = ["lei", "action_taken"]
required_lookup = ["LEI", "Entity.LegalName"]

for col in required_hmda:
    if col not in hmda.columns:
        raise ValueError(f"Missing '{col}' in HMDA dataset")

for col in required_lookup:
    if col not in lei_lookup.columns:
        raise ValueError(f"Missing '{col}' in LEI lookup dataset")

# -----------------------------------------
# KEEP ONLY NEEDED COLUMNS
# -----------------------------------------
hmda = hmda[["lei", "action_taken"]].copy()
lei_lookup = lei_lookup[["LEI", "Entity.LegalName"]].copy()

# -----------------------------------------
# CLEAN FORMATTING
# -----------------------------------------
hmda["lei"] = hmda["lei"].astype(str).str.strip().str.upper()
lei_lookup["LEI"] = lei_lookup["LEI"].astype(str).str.strip().str.upper()

hmda["action_taken"] = pd.to_numeric(hmda["action_taken"], errors="coerce")

# remove bad LEIs
hmda = hmda[
    hmda["lei"].notna() &
    (hmda["lei"] != "") &
    (hmda["lei"] != "NAN")
].copy()

lei_lookup = lei_lookup[
    lei_lookup["LEI"].notna() &
    (lei_lookup["LEI"] != "") &
    (lei_lookup["LEI"] != "NAN")
].copy()

# clean lookup names
lei_lookup["Entity.LegalName"] = (
    lei_lookup["Entity.LegalName"]
    .astype(str)
    .str.strip()
    .replace(["", "nan", "None", "NONE"], pd.NA)
)

# keep first non-null name for each LEI
lei_lookup = (
    lei_lookup.sort_values(by="Entity.LegalName", na_position="last")
    .drop_duplicates(subset=["LEI"], keep="first")
)

# -----------------------------------------
# KEEP ONLY APPROVALS AND DENIALS
# 1 = originated / approved
# 3 = denied
# -----------------------------------------
hmda = hmda[hmda["action_taken"].isin([1, 3])].copy()

# -----------------------------------------
# CREATE FLAGS
# -----------------------------------------
hmda["approved_flag"] = (hmda["action_taken"] == 1).astype(int)
hmda["denied_flag"] = (hmda["action_taken"] == 3).astype(int)

# -----------------------------------------
# GROUP BY LEI
# -----------------------------------------
lei_summary = (
    hmda.groupby("lei", as_index=False)
    .agg(
        approved_count=("approved_flag", "sum"),
        denied_count=("denied_flag", "sum")
    )
)

# -----------------------------------------
# CALCULATE TOTALS / RATES
# -----------------------------------------
lei_summary["total_apps"] = lei_summary["approved_count"] + lei_summary["denied_count"]

lei_summary = lei_summary[lei_summary["total_apps"] >= min_apps].copy()

lei_summary["approval_rate"] = lei_summary["approved_count"] / lei_summary["total_apps"]
lei_summary["denial_rate"] = lei_summary["denied_count"] / lei_summary["total_apps"]

lei_summary["approval_rate_pct"] = (lei_summary["approval_rate"] * 100).round(2)
lei_summary["denial_rate_pct"] = (lei_summary["denial_rate"] * 100).round(2)

# -----------------------------------------
# MERGE IN LEI NAMES
# -----------------------------------------
lei_summary = lei_summary.merge(
    lei_lookup,
    left_on="lei",
    right_on="LEI",
    how="left"
)

lei_summary.drop(columns=["LEI"], inplace=True)

# save unmatched before fill
unmatched_leis = lei_summary[lei_summary["Entity.LegalName"].isna()].copy()

lei_summary["Entity.LegalName"] = lei_summary["Entity.LegalName"].fillna("Unknown LEI Name")

# -----------------------------------------
# REORDER COLUMNS
# -----------------------------------------
lei_summary = lei_summary[
    [
        "lei",
        "Entity.LegalName",
        "approved_count",
        "denied_count",
        "total_apps",
        "approval_rate",
        "approval_rate_pct",
        "denial_rate",
        "denial_rate_pct"
    ]
].copy()

# -----------------------------------------
# CREATE RANKINGS
# -----------------------------------------
top_by_approvals = lei_summary.sort_values(
    by=["approved_count", "approval_rate"],
    ascending=[False, False]
).reset_index(drop=True)

top_by_denials = lei_summary.sort_values(
    by=["denied_count", "denial_rate"],
    ascending=[False, False]
).reset_index(drop=True)

top_by_approval_rate = lei_summary.sort_values(
    by=["approval_rate", "total_apps", "approved_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

top_by_denial_rate = lei_summary.sort_values(
    by=["denial_rate", "total_apps", "denied_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

# named-only versions
named_only = lei_summary[lei_summary["Entity.LegalName"] != "Unknown LEI Name"].copy()

named_top_by_approvals = named_only.sort_values(
    by=["approved_count", "approval_rate"],
    ascending=[False, False]
).reset_index(drop=True)

named_top_by_denials = named_only.sort_values(
    by=["denied_count", "denial_rate"],
    ascending=[False, False]
).reset_index(drop=True)

named_top_by_approval_rate = named_only.sort_values(
    by=["approval_rate", "total_apps", "approved_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

named_top_by_denial_rate = named_only.sort_values(
    by=["denial_rate", "total_apps", "denied_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

# -----------------------------------------
# SAVE OUTPUTS
# -----------------------------------------
os.makedirs(out_dir, exist_ok=True)

lei_summary.to_csv(os.path.join(out_dir, "lei_summary_from_raw_hmda.csv"), index=False)
top_by_approvals.to_csv(os.path.join(out_dir, "top_leis_by_approvals_from_raw_hmda.csv"), index=False)
top_by_denials.to_csv(os.path.join(out_dir, "top_leis_by_denials_from_raw_hmda.csv"), index=False)
top_by_approval_rate.to_csv(os.path.join(out_dir, "top_leis_by_approval_rate_from_raw_hmda.csv"), index=False)
top_by_denial_rate.to_csv(os.path.join(out_dir, "top_leis_by_denial_rate_from_raw_hmda.csv"), index=False)

named_only.to_csv(os.path.join(out_dir, "lei_summary_named_only_from_raw_hmda.csv"), index=False)
named_top_by_approvals.to_csv(os.path.join(out_dir, "named_top_leis_by_approvals_from_raw_hmda.csv"), index=False)
named_top_by_denials.to_csv(os.path.join(out_dir, "named_top_leis_by_denials_from_raw_hmda.csv"), index=False)
named_top_by_approval_rate.to_csv(os.path.join(out_dir, "named_top_leis_by_approval_rate_from_raw_hmda.csv"), index=False)
named_top_by_denial_rate.to_csv(os.path.join(out_dir, "named_top_leis_by_denial_rate_from_raw_hmda.csv"), index=False)

unmatched_leis.to_csv(os.path.join(out_dir, "unmatched_leis_from_raw_hmda.csv"), index=False)

# -----------------------------------------
# PRINT RESULTS
# -----------------------------------------
print(f"\nMinimum application threshold used: {min_apps}")
print(f"Final LEI summary shape: {lei_summary.shape}")
print(f"Named-only LEI summary shape: {named_only.shape}")
print(f"Unmatched LEIs: {unmatched_leis.shape[0]}")

print("\nTop 20 LEIs by APPROVAL COUNT:\n")
print(top_by_approvals.head(20).to_string(index=False))

print("\nTop 20 LEIs by DENIAL COUNT:\n")
print(top_by_denials.head(20).to_string(index=False))

print("\nTop 20 LEIs by APPROVAL RATE:\n")
print(top_by_approval_rate.head(20).to_string(index=False))

print("\nTop 20 LEIs by DENIAL RATE:\n")
print(top_by_denial_rate.head(20).to_string(index=False))

print("\nFiles saved in cleaned_data folder.")