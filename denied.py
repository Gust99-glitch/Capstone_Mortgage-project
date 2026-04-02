import pandas as pd
import numpy as np
import os

# Load combined HMDA dataset
mortgage_data = pd.read_csv("data_hmda_nc/nc_2019_2024.csv", low_memory=False)

# -----------------------------------------
# 1. Numeric cleanup
# -----------------------------------------
mortgage_data["action_taken"] = pd.to_numeric(
    mortgage_data["action_taken"], errors="coerce"
)

mortgage_data["loan_amount"] = pd.to_numeric(
    mortgage_data["loan_amount"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

mortgage_data["income"] = pd.to_numeric(
    mortgage_data["income"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

mortgage_data["property_value"] = pd.to_numeric(
    mortgage_data["property_value"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

mortgage_data["activity_year"] = pd.to_numeric(
    mortgage_data["activity_year"], errors="coerce"
)

# -----------------------------------------
# 2. Keep only denied applications
# -----------------------------------------
mortgage_data = mortgage_data[mortgage_data["action_taken"] == 3]

# -----------------------------------------
# 3. Remove commercial/business loans
# -----------------------------------------
mortgage_data = mortgage_data[
    ~mortgage_data["business_or_commercial_purpose"].isin([1, 1111, "1", "1111"])
]

# -----------------------------------------
# 4. Remove invalid demographic placeholders
# -----------------------------------------
mortgage_data = mortgage_data[
    ~mortgage_data["derived_race"].isin(["Race Not Available", "Free Form Text Only"])
]

mortgage_data = mortgage_data[
    ~mortgage_data["derived_ethnicity"].isin(["Ethnicity Not Available", "Free Form Text Only"])
]

mortgage_data = mortgage_data[
    mortgage_data["derived_sex"] != "Sex Not Available"
]

mortgage_data = mortgage_data[
    mortgage_data["applicant_age"] != 8888
]

# -----------------------------------------
# 5. Remove invalid debt_to_income_ratio values
# -----------------------------------------
mortgage_data = mortgage_data[
    ~mortgage_data["debt_to_income_ratio"].isin(["NA", "Exempt"])
]

# -----------------------------------------
# 6. Remove invalid conforming_loan_limit values
# -----------------------------------------
mortgage_data = mortgage_data[
    mortgage_data["conforming_loan_limit"] != "NA"
]

# -----------------------------------------
# 7. Map denial reason codes to readable labels
# -----------------------------------------
denial_reason_map = {
    1: "Debt-to-income ratio",
    2: "Employment history",
    3: "Credit history",
    4: "Collateral",
    5: "Insufficient cash",
    6: "Unverifiable information",
    7: "Credit application incomplete",
    8: "Mortgage insurance denied",
    9: "Other",
    10: "Not applicable"
}

for col in ["denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4"]:
    if col in mortgage_data.columns:
        mortgage_data[col] = pd.to_numeric(mortgage_data[col], errors="coerce")
        mortgage_data[col + "_label"] = mortgage_data[col].map(denial_reason_map)

# -----------------------------------------
# 8. Keep only the columns that should NOT be null for denied loans
# -----------------------------------------
NO_NULL_COLS = [
    "lei",
    "action_taken",
    "loan_amount",
    "income",
    "property_value",
    "loan_term",
    "loan_purpose",
    "loan_type",
    "occupancy_type",
    "derived_loan_product_type",
    "derived_dwelling_category",
    "county_code",
    "derived_msa-md",
    "activity_year",
    "tract_to_msa_income_percentage",
    "tract_population",
    "tract_minority_population_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
    "denial_reason-1"
]

# Only enforce rules for columns that actually exist
existing_no_null_cols = [c for c in NO_NULL_COLS if c in mortgage_data.columns]

mortgage_data = mortgage_data.dropna(subset=existing_no_null_cols)

# -----------------------------------------
# 9. Add readable action_taken label
# -----------------------------------------
mortgage_data["action_taken_label"] = "Application denied"

# -----------------------------------------
# 10. Reset index
# -----------------------------------------
mortgage_data.reset_index(drop=True, inplace=True)

# -----------------------------------------
# 11. Save cleaned denied-loans dataset
# -----------------------------------------
OUT_DIR = "cleaned_data"
os.makedirs(OUT_DIR, exist_ok=True)

cleaned_path = os.path.join(OUT_DIR, "nc_2019_2024_denied_cleaned.csv")
mortgage_data.to_csv(cleaned_path, index=False)

print("Final denied dataset shape:", mortgage_data.shape)
print(mortgage_data[[
    "activity_year",
    "action_taken",
    "action_taken_label",
    "loan_amount",
    "income",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "denial_reason-1",
    "denial_reason-1_label"
]].head())