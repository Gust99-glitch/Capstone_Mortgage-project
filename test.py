
# Change this to your actual CSV file name
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------
# FILE PATH
# ---------------------------------
file_path = r"C:\Users\Carlin Crawford\Desktop\DTSC 4301\Capstone_Mortgage-project\data_hmda_nc\nc_2019_2024.csv"

# ---------------------------------
# LOAD DATA
# ---------------------------------
df = pd.read_csv(file_path, low_memory=False)

# ---------------------------------
# ACTION TAKEN LABELS
# ---------------------------------
action_labels = {
    1: "Loan originated",
    2: "Application approved but not accepted",
    3: "Application denied",
    4: "Application withdrawn by applicant",
    5: "File closed for incompleteness",
    6: "Purchased loan",
    7: "Preapproval request denied",
    8: "Preapproval request approved but not accepted"
}

years_to_keep = [2019, 2020, 2021, 2022, 2023, 2024]

# ---------------------------------
# CLEAN COLUMNS
# ---------------------------------
df["activity_year"] = pd.to_numeric(df["activity_year"], errors="coerce")
df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")

# keep only needed years and valid action codes
df = df[
    df["activity_year"].isin(years_to_keep) &
    df["action_taken"].isin(action_labels.keys())
].copy()

# output folder
output_folder = Path("action_taken_by_year_outputs")
output_folder.mkdir(exist_ok=True)

# ---------------------------------
# YEAR-BY-YEAR DISTRIBUTION
# ---------------------------------
for year in years_to_keep:
    year_df = df[df["activity_year"] == year].copy()

    counts = (
        year_df["action_taken"]
        .value_counts()
        .reindex(range(1, 9), fill_value=0)
    )

    percentages = (counts / counts.sum() * 100).round(2)

    distribution_table = pd.DataFrame({
        "Activity Year": year,
        "Action Code": counts.index,
        "Outcome": [action_labels[i] for i in counts.index],
        "Count": counts.values,
        "Percent": percentages.values
    })

    print(f"\n{'='*70}")
    print(f"ACTION TAKEN DISTRIBUTION FOR {year}")
    print(f"{'='*70}")
    print(distribution_table.to_string(index=False))

    # save csv
    csv_path = output_folder / f"action_taken_distribution_{year}.csv"
    distribution_table.to_csv(csv_path, index=False)

    # bar chart
    plt.figure(figsize=(14, 7))
    plt.bar(distribution_table["Outcome"], distribution_table["Count"])
    plt.title(f"Distribution of Loan Application Outcomes ({year})", fontsize=14)
    plt.xlabel("Outcome", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path = output_folder / f"action_taken_distribution_{year}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------
# COMBINED TABLE FOR ALL YEARS
# ---------------------------------
combined_counts = (
    df.groupby(["activity_year", "action_taken"])
      .size()
      .unstack(fill_value=0)
      .reindex(index=years_to_keep, columns=range(1, 9), fill_value=0)
)

combined_counts.columns = [action_labels[col] for col in combined_counts.columns]
combined_counts.index.name = "Activity Year"

combined_csv_path = output_folder / "action_taken_distribution_all_years.csv"
combined_counts.to_csv(combined_csv_path)

print(f"\nSaved all outputs in folder: {output_folder.resolve()}")
print("Includes:")
print("- 1 CSV per year")
print("- 1 PNG chart per year")
print("- 1 combined CSV for all years")