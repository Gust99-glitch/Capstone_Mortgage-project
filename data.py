import pandas as pd
import numpy as np
import os

# Load combined HMDA dataset
mortgage_data = pd.read_csv("data_hmda_nc/nc_2019_2024.csv", low_memory=False)

# -----------------------------------------
# 1. Numeric cleanup
# -----------------------------------------
mortgage_data["loan_amount"] = pd.to_numeric(
    mortgage_data["loan_amount"].astype(str).str.replace(",", ""),
    errors="coerce"
)

mortgage_data["interest_rate"] = pd.to_numeric(
    mortgage_data["interest_rate"], errors="coerce"
)

mortgage_data["applicant_age"] = pd.to_numeric(
    mortgage_data["applicant_age"], errors="coerce"
)

# -----------------------------------------
# 2. Remove commercial/business loans
# -----------------------------------------
mortgage_data = mortgage_data[
    ~mortgage_data["business_or_commercial_purpose"].isin([1, 1111])
]

# -----------------------------------------
# 3. Remove invalid demographic placeholders
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
# 4. Remove invalid debt_to_income_ratio values
# -----------------------------------------
mortgage_data = mortgage_data[
    ~mortgage_data["debt_to_income_ratio"].isin(["NA", "Exempt"])
]

# -----------------------------------------
# 5. Remove invalid conforming_loan_limit values
# -----------------------------------------
mortgage_data = mortgage_data[
    mortgage_data["conforming_loan_limit"] != "NA"
]

# -----------------------------------------
# 6. Drop rows where ANY “no-null” column is missing
# -----------------------------------------
NO_NULL_COLS = [
    "lei",
    "action_taken",
    "interest_rate",
    "loan_amount",
    "rate_spread",
    "total_loan_costs",
    "discount_points",
    "income",
    "property_value",
    "lender_credits",
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
    "tract_median_age_of_housing_units"
]

# Only enforce rules for columns that actually exist
existing_no_null_cols = [c for c in NO_NULL_COLS if c in mortgage_data.columns]

mortgage_data = mortgage_data.dropna(subset=existing_no_null_cols)

# -----------------------------------------
# Reset index
# -----------------------------------------
mortgage_data.reset_index(drop=True, inplace=True)

# -----------------------------------------
# Save cleaned dataset
# -----------------------------------------
OUT_DIR = "cleaned_data"
os.makedirs(OUT_DIR, exist_ok=True)

cleaned_path = os.path.join(OUT_DIR, "nc_2019_2024_cleaned.csv")
mortgage_data.to_csv(cleaned_path, index=False)

print("Final dataset shape:", mortgage_data.shape)
print(mortgage_data.head())