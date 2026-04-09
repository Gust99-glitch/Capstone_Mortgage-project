import os
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
STATE = "NC"
YEARS = range(2019, 2025)  # 2019–2024
OUT_DIR = "data_hmda_nc"

os.makedirs(OUT_DIR, exist_ok=True)

# The 35 columns we originally hand-picked (kept here for reference only)
# Feature selection will now be done by the model on ALL available columns
KEEP_COLS_ORIGINAL = [
    "lei", "action_taken", "interest_rate", "loan_amount", "rate_spread",
    "hoepa_status", "total_loan_costs", "discount_points", "income",
    "debt_to_income_ratio", "property_value", "lender_credits", "denial_reason-1",
    "loan_term", "loan_purpose", "loan_type", "occupancy_type",
    "derived_loan_product_type", "derived_dwelling_category", "conforming_loan_limit",
    "county_code", "derived_msa-md", "activity_year", "applicant_age",
    "derived_race", "derived_ethnicity", "derived_sex", "business_or_commercial_purpose",
    "tract_to_msa_income_percentage", "tract_population", "tract_minority_population_percentage",
    "ffiec_msa_md_median_family_income", "tract_owner_occupied_units",
    "tract_one_to_four_family_homes", "tract_median_age_of_housing_units",
]

# Set to None to keep ALL columns from the API (recommended for model-based feature selection)
# Set to KEEP_COLS_ORIGINAL to revert to the original 35-column subset
KEEP_COLS = None
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

def download_csv(year: int, path: str, retries: int = 5) -> None:
    """
    Downloads HMDA CSV data for a given year using streaming
    to avoid loading everything into memory. Retries on failure.
    """
    url = api_csv_url(year)
    for attempt in range(1, retries + 1):
        print(f"\nDownloading {year} (attempt {attempt})...")
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {year}",
                    ncols=80,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return  # success
        except Exception as e:
            print(f"  Error: {e}")
            if attempt == retries:
                raise
            import time
            time.sleep(5 * attempt)

# ---------------------------
# Cleaning
# ---------------------------
def clean_one_year(csv_path: str) -> pd.DataFrame:
    """
    Reads a raw HMDA CSV file in chunks, keeps selected columns,
    and applies basic numeric cleaning.
    """
    chunks = []
    file_size = os.path.getsize(csv_path)

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
              desc="  Cleaning", ncols=80) as bar:
        for df in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
            if KEEP_COLS is not None:
                keep = [c for c in KEEP_COLS if c in df.columns]
                df = df[keep]
            chunks.append(df)
            bar.update(df.memory_usage(deep=True).sum())

    out = pd.concat(chunks, ignore_index=True)

    # Numeric cleanup
    if "loan_amount" in out.columns:
        out["loan_amount"] = pd.to_numeric(out["loan_amount"], errors="coerce")

    if "interest_rate" in out.columns:
        out["interest_rate"] = pd.to_numeric(out["interest_rate"], errors="coerce")

    

    return out

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    all_years = []

    for year in YEARS:
        raw_csv = os.path.join(OUT_DIR, f"nc_{year}_raw.csv")
        clean_csv = os.path.join(OUT_DIR, f"nc_{year}.csv")

        # Skip if already cleaned
        if os.path.exists(clean_csv):
            print(f"Skipping {year} (already done)")
            all_years.append(pd.read_csv(clean_csv, low_memory=False))
            continue

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