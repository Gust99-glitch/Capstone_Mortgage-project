import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------------
# FILE PATHS
# -----------------------------
csv_path = r"C:\Users\Carlin Crawford\Downloads\postal_code_summary1 (1).csv"
zcta_path = r"C:\Users\Carlin Crawford\Downloads\tl_2020_us_zcta520\tl_2020_us_zcta520.shp"
county_path = r"C:\Users\Carlin Crawford\Downloads\tl_2020_us_county\tl_2020_us_county.shp"

# -----------------------------
# 1) LOAD CSV
# -----------------------------
df = pd.read_csv(csv_path)

# keep only NC if the column exists
if "Reigon_Clean" in df.columns:
    df = df[df["Reigon_Clean"] == "US-NC"].copy()

df = df[["Entity.HeadquartersAddress.PostalCode", "Company_Count"]].copy()

df["ZIP5"] = (
    df["Entity.HeadquartersAddress.PostalCode"]
    .astype(str)
    .str.strip()
    .str[:5]
    .str.zfill(5)
)

df = df[df["ZIP5"].str.match(r"^\d{5}$", na=False)].copy()
df = df.groupby("ZIP5", as_index=False)["Company_Count"].sum()

print("CSV loaded.")
print(df.head())

# -----------------------------
# 2) LOAD ZCTA SHAPEFILE
# -----------------------------
zcta = gpd.read_file(zcta_path)
zcta["ZIP5"] = zcta["ZCTA5CE20"].astype(str).str.zfill(5)

print("ZCTA shapefile loaded.")
print(zcta[["ZIP5"]].head())

# -----------------------------
# 3) JOIN ZIP COUNTS TO POLYGONS
# -----------------------------
gdf = zcta.merge(df, on="ZIP5", how="inner")

print("Joined ZIP polygons with company counts.")
print(gdf[["ZIP5", "Company_Count"]].head())

# -----------------------------
# 4) FILTER TO NC BOUNDING BOX
# -----------------------------
minx, miny, maxx, maxy = -84.5, 33.5, -75.0, 36.8
gdf = gdf.cx[minx:maxx, miny:maxy]

print("Filtered to NC bounding box.")

# -----------------------------
# 5) LOAD COUNTY BOUNDARIES
# -----------------------------
counties = gpd.read_file(county_path)
counties_nc = counties[counties["STATEFP"] == "37"].copy()
counties_nc = counties_nc.to_crs(gdf.crs)

print("County shapefile loaded.")
print(counties_nc.head())

# -----------------------------
# 6) PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 12))

gdf.plot(
    column="Company_Count",
    cmap="viridis",
    linewidth=0.1,
    edgecolor="gray",
    legend=True,
    ax=ax
)

counties_nc.boundary.plot(
    ax=ax,
    linewidth=0.7
)

ax.set_title("North Carolina Postal Code Frequencies with County Lines", fontsize=14)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("nc_postal_heatmap_with_counties.png", dpi=300, bbox_inches="tight")
plt.show()

print("Map saved as nc_postal_heatmap_with_counties.png")