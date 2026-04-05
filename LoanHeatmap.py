import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =========================================================
# FILE PATHS
# =========================================================
csv_path = "data/Countycode_Loans_Rate.csv"
county_path = "data/tl_2020_us_county/tl_2020_us_county.shp"

# =========================================================
# LOAD COUNTY-LEVEL LOAN DATA
# =========================================================
df = pd.read_csv(csv_path, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# Keep only needed columns
df = df[["county_code", "total_loans", "avg_loan", "avg_rate"]].copy()

# Clean fields
df["county_code"] = df["county_code"].astype(str).str.strip().str.zfill(5)
df["total_loans"] = pd.to_numeric(df["total_loans"], errors="coerce")
df["avg_loan"] = pd.to_numeric(df["avg_loan"], errors="coerce")
df["avg_rate"] = pd.to_numeric(df["avg_rate"], errors="coerce")

# Keep only NC counties
df = df[df["county_code"].str.startswith("37")].copy()
df = df.dropna(subset=["county_code", "total_loans", "avg_loan", "avg_rate"])

# =========================================================
# LOAD COUNTY SHAPEFILE
# =========================================================
counties = gpd.read_file(county_path)

# Keep only NC counties using state FIPS = 37
counties_nc = counties[counties["STATEFP"] == "37"].copy()

# Build full 5-digit county FIPS for join
counties_nc["county_code"] = (
    counties_nc["STATEFP"].astype(str).str.zfill(2)
    + counties_nc["COUNTYFP"].astype(str).str.zfill(3)
)

# Join data to county polygons
gdf = counties_nc.merge(df, on="county_code", how="left")

# Optional: drop counties with no data
gdf = gdf.dropna(subset=["total_loans", "avg_loan", "avg_rate"]).copy()

# =========================================================
# HELPER FUNCTION TO DRAW AND SAVE A MAP
# =========================================================
def make_map(data, column, title, filename, cmap="viridis", legend_label=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    data.plot(
        column=column,
        cmap=cmap,
        linewidth=0.6,
        edgecolor="black",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgray", "label": "No data"}
    )

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# MAKE THREE SEPARATE MAPS
# =========================================================
make_map(
    gdf,
    column="total_loans",
    title="North Carolina Counties - Total Loans",
    filename="nc_total_loans_map.png",
    cmap="Blues"
)

make_map(
    gdf,
    column="avg_loan",
    title="North Carolina Counties - Average Loan Amount",
    filename="nc_avg_loan_map.png",
    cmap="Greens"
)

make_map(
    gdf,
    column="avg_rate",
    title="North Carolina Counties - Average Interest Rate",
    filename="nc_avg_rate_map.png",
    cmap="Reds"
)

# =========================================================
# OPTIONAL: ONE 3-PANEL IMAGE TOO
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

gdf.plot(column="total_loans", cmap="Blues", linewidth=0.6, edgecolor="black", legend=True, ax=axes[0])
axes[0].set_title("Total Loans")
axes[0].set_axis_off()

gdf.plot(column="avg_loan", cmap="Greens", linewidth=0.6, edgecolor="black", legend=True, ax=axes[1])
axes[1].set_title("Average Loan Amount")
axes[1].set_axis_off()

gdf.plot(column="avg_rate", cmap="Reds", linewidth=0.6, edgecolor="black", legend=True, ax=axes[2])
axes[2].set_title("Average Interest Rate")
axes[2].set_axis_off()

plt.tight_layout()
plt.savefig("nc_3panel_county_maps.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved:")
print(" - nc_total_loans_map.png")
print(" - nc_avg_loan_map.png")
print(" - nc_avg_rate_map.png")
print(" - nc_3panel_county_maps.png")