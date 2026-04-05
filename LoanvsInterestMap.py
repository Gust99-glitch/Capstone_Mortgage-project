import pandas as pd
import plotly.express as px

file_path = "data/Countycode_Loans_Rate.csv"

df = pd.read_csv(file_path, low_memory=False)

df.columns = df.columns.str.strip().str.lower()

df = df[["county_code", "total_loans", "avg_loan", "avg_rate"]].copy()

df["county_code"] = df["county_code"].astype(str).str.strip().str.zfill(5)
df["total_loans"] = pd.to_numeric(df["total_loans"], errors="coerce")
df["avg_loan"] = pd.to_numeric(df["avg_loan"], errors="coerce")
df["avg_rate"] = pd.to_numeric(df["avg_rate"], errors="coerce")

df = df[df["county_code"].str.startswith("37")].copy()
df = df.dropna(subset=["county_code", "total_loans", "avg_loan", "avg_rate"])
df = df[(df["avg_loan"] > 0) & (df["avg_rate"] > 0)]

nc_county_names = {
    "37001": "Alamance", "37003": "Alexander", "37005": "Alleghany", "37007": "Anson",
    "37009": "Ashe", "37011": "Avery", "37013": "Beaufort", "37015": "Bertie",
    "37017": "Bladen", "37019": "Brunswick", "37021": "Buncombe", "37023": "Burke",
    "37025": "Cabarrus", "37027": "Caldwell", "37029": "Camden", "37031": "Carteret",
    "37033": "Caswell", "37035": "Catawba", "37037": "Chatham", "37039": "Cherokee",
    "37041": "Chowan", "37043": "Clay", "37045": "Cleveland", "37047": "Columbus",
    "37049": "Craven", "37051": "Cumberland", "37053": "Currituck", "37055": "Dare",
    "37057": "Davidson", "37059": "Davie", "37061": "Duplin", "37063": "Durham",
    "37065": "Edgecombe", "37067": "Forsyth", "37069": "Franklin", "37071": "Gaston",
    "37073": "Gates", "37075": "Graham", "37077": "Granville", "37079": "Greene",
    "37081": "Guilford", "37083": "Halifax", "37085": "Harnett", "37087": "Haywood",
    "37089": "Henderson", "37091": "Hertford", "37093": "Hoke", "37095": "Hyde",
    "37097": "Iredell", "37099": "Jackson", "37101": "Johnston", "37103": "Jones",
    "37105": "Lee", "37107": "Lenoir", "37109": "Lincoln", "37111": "McDowell",
    "37113": "Macon", "37115": "Madison", "37117": "Martin", "37119": "Mecklenburg",
    "37121": "Mitchell", "37123": "Montgomery", "37125": "Moore", "37127": "Nash",
    "37129": "New Hanover", "37131": "Northampton", "37133": "Onslow", "37135": "Orange",
    "37137": "Pamlico", "37139": "Pasquotank", "37141": "Pender", "37143": "Perquimans",
    "37145": "Person", "37147": "Pitt", "37149": "Polk", "37151": "Randolph",
    "37153": "Richmond", "37155": "Robeson", "37157": "Rockingham", "37159": "Rowan",
    "37161": "Rutherford", "37163": "Sampson", "37165": "Scotland", "37167": "Stanly",
    "37169": "Stokes", "37171": "Surry", "37173": "Swain", "37175": "Transylvania",
    "37177": "Tyrrell", "37179": "Union", "37181": "Vance", "37183": "Wake",
    "37185": "Warren", "37187": "Washington", "37189": "Watauga", "37191": "Wayne",
    "37193": "Wilkes", "37195": "Wilson", "37197": "Yadkin", "37199": "Yancey"
}

df["county_name"] = df["county_code"].map(nc_county_names).fillna(df["county_code"])

fig = px.scatter(
    df,
    x="avg_rate",
    y="avg_loan",
    size="total_loans",
    hover_name="county_name",
    hover_data={
        "county_code": True,
        "total_loans": True,
        "avg_rate": ":.2f",
        "avg_loan": ":,.0f"
    },
    title="North Carolina Counties: Average Loan Size vs Average Interest Rate",
    labels={
        "avg_rate": "Average Interest Rate (%)",
        "avg_loan": "Average Loan Amount ($)",
        "total_loans": "Total Loans"
    },
    size_max=60
)

fig.update_layout(title_x=0.5)

fig.show()
fig.write_image("nc_county_loans_rate_scatter.png", width=1200, height=800)

print("Saved: nc_county_loans_rate_scatter.png")