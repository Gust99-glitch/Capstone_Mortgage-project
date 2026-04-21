"""
important_feature.py — Download NC HMDA Data (30 Lasso Features Only)
=======================================================================

Downloads NC HMDA data from the FFIEC API for each year and keeps
only the 23 base columns that produce the 30 Lasso-selected features.

Output:
  • data_hmda_nc/nc_lasso_features.csv  — combined data, 30 columns only
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
STATE   = "NC"
YEARS   = range(2019, 2025)   # 2019–2024
OUT_DIR = "data_hmda_nc"

os.makedirs(OUT_DIR, exist_ok=True)

# The 23 raw columns that map to the 30 Lasso-selected features after encoding
# (e.g. derived_race → derived_race_White, derived_race_Asian, etc.)
LASSO_BASE_COLS = [
    "action_taken",
    "applicant_credit_score_type",
    "preapproval",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_purpose",
    "lien_status",
    "loan_term",
    "loan_to_value_ratio",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "prepayment_penalty_term",
    "intro_rate_period",
    "derived_race",
    "derived_dwelling_category",
    "applicant_age",
    "co-applicant_age",
    "co-applicant_sex",
    "co-applicant_sex_observed",
    "co-applicant_ethnicity_observed",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
]

# ---------------------------
# API helpers
# ---------------------------
def api_csv_url(year: int) -> str:
    return (
        "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
        f"?states={STATE}&years={year}"
    )


def download_year(year: int, retries: int = 5) -> pd.DataFrame:
    """
    Downloads one year of NC HMDA data from the FFIEC API,
    keeps only the Lasso base columns, and returns a DataFrame.
    Raw file is deleted after parsing.
    """
    url      = api_csv_url(year)
    raw_path = os.path.join(OUT_DIR, f"nc_{year}_raw.csv")

    for attempt in range(1, retries + 1):
        print(f"\nDownloading {year} (attempt {attempt})...")
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(raw_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=f"  {year}", ncols=80,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            break
        except Exception as e:
            print(f"  Error: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    df = pd.read_csv(
        raw_path,
        usecols=lambda c: c in LASSO_BASE_COLS,
        low_memory=False,
    )
    os.remove(raw_path)
    print(f"  {year}: {len(df):,} rows, {df.shape[1]} columns kept")
    return df


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print("=" * 55)
    print("  Downloading NC HMDA — Lasso Features Only")
    print("=" * 55)

    frames = []
    for year in YEARS:
        frames.append(download_year(year))

    combined = pd.concat(frames, ignore_index=True)

    out_path = os.path.join(OUT_DIR, "nc_lasso_features.csv")
    combined.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total rows   : {len(combined):,}")
    print(f"Total columns: {combined.shape[1]}")
    print(f"\nColumns included:")
    for col in combined.columns:
        print(f"  • {col}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()
