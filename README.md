# Capstone Mortgage Project

A data pipeline for downloading, cleaning, and analyzing Home Mortgage Disclosure Act (HMDA) data for North Carolina (2019–2024).

## Overview

This project retrieves publicly available HMDA loan-level data from the FFIEC API, cleans it, and produces a structured dataset ready for mortgage analysis — including approval/denial patterns, interest rates, borrower demographics, and lender identifiers.

## Project Structure

```
.
├── mortgage.py       # Downloads and cleans raw HMDA data by year
├── data.py           # Applies additional filtering and produces final dataset
├── data_hmda_nc/     # Per-year cleaned CSVs + combined (gitignored)
└── cleaned_data/     # Final analysis-ready dataset (gitignored)
```

## Data Source

- **FFIEC HMDA Data Browser API** — `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv`
- State: North Carolina (`NC`)
- Years: 2019–2024

## Pipeline

### Step 1 — `mortgage.py`

- Downloads raw HMDA CSVs for each year via the FFIEC API (streaming, with retry logic)
- Keeps a curated set of columns (lender LEI, loan features, borrower demographics, census tract context)
- Saves a cleaned per-year CSV and deletes the raw file to save disk space
- Combines all years into `data_hmda_nc/nc_2019_2024.csv`

### Step 2 — `data.py`

- Loads the combined dataset
- Removes commercial/business loans
- Removes rows with invalid demographic placeholders (`Race Not Available`, etc.)
- Removes invalid `debt_to_income_ratio` and `conforming_loan_limit` values
- Drops rows with missing values in required columns
- Saves the final dataset to `cleaned_data/nc_2019_2024_cleaned.csv`

## Columns Kept

| Category | Columns |
|---|---|
| Lender | `lei` |
| Target / Outcomes | `action_taken`, `interest_rate`, `loan_amount`, `rate_spread`, `hoepa_status`, `total_loan_costs`, `discount_points` |
| Borrower | `income`, `debt_to_income_ratio`, `property_value`, `lender_credits`, `applicant_age`, `denial_reason-1` |
| Loan | `loan_term`, `loan_purpose`, `loan_type`, `occupancy_type`, `derived_loan_product_type`, `derived_dwelling_category`, `conforming_loan_limit` |
| Demographics | `derived_race`, `derived_ethnicity`, `derived_sex` |
| Geography | `county_code`, `derived_msa-md`, `activity_year` |
| Census Tract | `tract_to_msa_income_percentage`, `tract_population`, `tract_minority_population_percentage`, `ffiec_msa_md_median_family_income`, `tract_owner_occupied_units`, `tract_one_to_four_family_homes`, `tract_median_age_of_housing_units` |

## Usage

```bash
# Install dependencies
pip install requests pandas numpy

# Step 1: Download and clean raw data
python mortgage.py

# Step 2: Apply final cleaning and save analysis-ready dataset
python data.py
```

The final dataset is saved to `cleaned_data/nc_2019_2024_cleaned.csv`.

## Output

- **Raw combined**: `data_hmda_nc/nc_2019_2024.csv` — ~3.97M rows
- **Cleaned final**: `cleaned_data/nc_2019_2024_cleaned.csv` — ~143K rows, 35 columns
