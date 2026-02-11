import os
import requests
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
STATE = "NC"
YEARS = range(2019, 2025)  # 2019–2024
OUT_DIR = "data_hmda_nc"

os.makedirs(OUT_DIR, exist_ok=True)

# Columns we care about (only keep these if they exist)
KEEP_COLS = [
    # target/outcomes
    "action_taken",          # approval/denial
    "interest_rate",
    "loan_amount",
    "rate_spread",          # interest rate spread (proxy for risk)
    "hoepa_status",     
    "total_loan_costs",     # proxy for fees
    "discount_points",       # proxy for fees

    # user-provided-ish features (or close proxies)
    "income",                        # thousands of dollars
    "debt_to_income_ratio",          # binned categories                                
    "property_value",
    "lender_credits",
    "denial_reason-1",                 # top reason for denial (if denied)
   
    # context features
    "loan_term",
    "loan_purpose",
    "loan_type",
    "occupancy_type",
    "derived_loan_product_type",
    "derived_dwelling_category",
    "conforming_loan_limit",

    # geography/time/demographics
    "county_code",
    "derived_msa-md",
    "activity_year",
    "ageapplicant",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "business_or_commercial_purpose",
   
   # census tract-level features (proxies for neighborhood context)
    "tract_to_msa_income_percentage",
    "tract_population",
    "tract_minority_population_percentage",
    "ffiec_msa_md_median_family_income",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",

]
# ---------------------------
# API helpers
# ---------------------------
def api_csv_url(year: int) -> str:
    """
    Returns the FFIEC HMDA API URL for NC data for a given year.
    """
    return (
        "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
        f"?states={STATE}&years={year}"
    )

def download_csv(year: int, path: str) -> None:
    """
    Downloads HMDA CSV data for a given year using streaming
    to avoid loading everything into memory.
    """
    url = api_csv_url(year)
    print(f"Downloading {year}...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

# ---------------------------
# Cleaning
# ---------------------------
def clean_one_year(csv_path: str) -> pd.DataFrame:
    """
    Reads a raw HMDA CSV file in chunks, keeps selected columns,
    and applies basic numeric cleaning.
    """
    chunks = []

    for df in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
        keep = [c for c in KEEP_COLS if c in df.columns]
        if keep:
            df = df[keep]
        chunks.append(df)

    out = pd.concat(chunks, ignore_index=True)

    # Numeric cleanup
    if "loan_amount" in out.columns:
        out["loan_amount"] = pd.to_numeric(out["loan_amount"], errors="coerce")

    if "interest_rate" in out.columns:
        out["interest_rate"] = pd.to_numeric(out["interest_rate"], errors="coerce")

    if "applicant_age" in out.columns:
        out["applicant_age"] = pd.to_numeric(out["applicant_age"], errors="coerce")

    return out

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    all_years = []

    for year in YEARS:
        raw_csv = os.path.join(OUT_DIR, f"nc_{year}_raw.csv")
        clean_csv = os.path.join(OUT_DIR, f"nc_{year}.csv")

        # 1) Download
        download_csv(year, raw_csv)

        # 2) Clean
        df = clean_one_year(raw_csv)
        df["year"] = year

        # 3) Save cleaned CSV
        df.to_csv(clean_csv, index=False)
        print(f"Saved {clean_csv} | rows: {len(df):,}")

        # 4) Delete raw CSV
        os.remove(raw_csv)

        all_years.append(df)

    # Optional: combine all years
    combined = pd.concat(all_years, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "nc_2019_2024.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\nCombined file saved: {combined_path}")
    print(f"Total rows: {len(combined):,}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()