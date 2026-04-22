import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mortgage Markets in Motion",
    page_icon="🏠",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf         = joblib.load("models/rf_model.joblib")
    rf_feats   = joblib.load("models/rf_feature_names.joblib")
    rf_meds    = joblib.load("models/rf_train_medians.joblib")
    fair       = joblib.load("models/fair_model.joblib")
    fair_feats = joblib.load("models/fair_feature_names.joblib")
    fair_meds  = joblib.load("models/fair_train_medians.joblib")
    return rf, rf_feats, rf_meds, fair, fair_feats, fair_meds

rf, rf_feats, rf_meds, fair, fair_feats, fair_meds = load_models()

RF_THRESHOLD   = 0.596
FAIR_THRESHOLD = 0.570

DTI_MAP = {v: i for i, v in enumerate([
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
])}

DEMOGRAPHIC_COLS = [
    "derived_race", "applicant_age", "co-applicant_age",
    "co-applicant_sex", "co-applicant_sex_observed",
    "co-applicant_ethnicity_observed",
]

MINORITY_RACES = [
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
]

# ─────────────────────────────────────────────
# LOAD LENDER FAIRNESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_lenders():
    path = "lender_fairness.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

lender_df = load_lenders()

# ─────────────────────────────────────────────
# HELPER — PREPROCESS & PREDICT
# ─────────────────────────────────────────────
def build_profile(inputs):
    return {
        "derived_dwelling_category"                : inputs["dwelling"],
        "derived_race"                             : inputs["race"],
        "preapproval"                              : 2,
        "loan_purpose"                             : inputs["loan_purpose"],
        "lien_status"                              : 1,
        "reverse_mortgage"                         : 2,
        "open-end_line_of_credit"                  : 2,
        "loan_amount"                              : inputs["loan_amount"],
        "loan_to_value_ratio"                      : inputs["ltv"],
        "loan_term"                                : inputs["loan_term"],
        "prepayment_penalty_term"                  : -1,
        "intro_rate_period"                        : -1,
        "manufactured_home_secured_property_type"  : 1111,
        "manufactured_home_land_property_interest" : 1111,
        "debt_to_income_ratio"                     : inputs["dti"],
        "applicant_credit_score_type"              : inputs["credit_score_type"],
        "co-applicant_ethnicity_observed"          : 3 if not inputs["has_co_applicant"] else 2,
        "co-applicant_sex"                         : 5 if not inputs["has_co_applicant"] else 2,
        "co-applicant_sex_observed"                : 3 if not inputs["has_co_applicant"] else 2,
        "ffiec_msa_md_median_family_income"        : inputs["area_income"],
        "tract_to_msa_income_percentage"           : inputs["tract_pct"],
        "applicant_age"                            : inputs["age"],
        "co-applicant_age"                         : "35-44" if inputs["has_co_applicant"] else "9999",
    }

def preprocess(profile):
    df = pd.DataFrame([profile])
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(DTI_MAP)
    cat_cols = ["derived_dwelling_category", "derived_race", "applicant_age", "co-applicant_age"]
    df = pd.get_dummies(df, columns=cat_cols, dtype=np.int8)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def align(df, feats, meds):
    X = df.reindex(columns=feats, fill_value=0)
    X = X.fillna(meds).fillna(0)
    return X

def predict(inputs):
    profile = build_profile(inputs)
    df      = preprocess(profile)

    X_rf   = align(df, rf_feats, rf_meds)
    rf_prob = float(rf.predict_proba(X_rf)[0, 1])

    df_fair = df.drop(columns=[c for c in df.columns
                                if any(c == d or c.startswith(d + "_")
                                       for d in DEMOGRAPHIC_COLS)], errors="ignore")
    X_fair     = align(df_fair, fair_feats, fair_meds)
    fair_prob  = float(fair.predict_proba(X_fair)[0, 1])

    rf_decision   = rf_prob   >= RF_THRESHOLD
    fair_decision = fair_prob >= FAIR_THRESHOLD
    bias_gap      = (fair_prob - rf_prob) * 100

    return rf_prob, rf_decision, fair_prob, fair_decision, bias_gap

def get_lender_recommendations(race, top_n=5):
    if lender_df is None:
        return None

    race_config = {
        "Black or African American"               : ("black_gap",  "black_rate",  "black_count",  100),
        "Asian"                                   : ("asian_gap",  "asian_rate",  "asian_count",  30),
        "American Indian or Alaska Native"        : ("native_gap", "native_rate", "native_count", 30),
        "Native Hawaiian or Other Pacific Islander": ("native_gap", "native_rate", "native_count", 30),
    }

    config = race_config.get(race)
    if not config:
        return None

    gap_col, rate_col, count_col, min_count = config

    if gap_col not in lender_df.columns or count_col not in lender_df.columns:
        return None

    filtered = lender_df[
        (lender_df[gap_col].notna()) &
        (lender_df[count_col] >= min_count)
    ].copy()

    filtered = filtered.sort_values(gap_col)

    cols = ["institution_name", "city", "state", "total_loans", "white_rate", rate_col, gap_col]
    cols = [c for c in cols if c in filtered.columns]
    return filtered[cols].head(top_n).reset_index(drop=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
st.title("🏠 Mortgage Markets in Motion")

HeatMapTab, UserInterfaceTab, ChartsTab = st.tabs([
    "📍 NC Heatmap",
    "💭 Predictive Tool",
    "📈 Research Charting",
])

# ══════════════════════════════════════════════
# TAB 1 — HEATMAP
# ══════════════════════════════════════════════
with HeatMapTab:
    st.header("A Geographic View of NC Loans")

    maps = {
        "Total Loans by County"       : "charts/nc_total_loans_map.png",
        "Average Loan Amount by County": "charts/nc_avg_loan_map.png",
        "Average Interest Rate by County": "charts/nc_avg_rate_map.png",
        "3-Panel County Overview"     : "charts/nc_3panel_county_maps.png",
        "Postal Heatmap with Counties": "charts/nc_postal_heatmap_with_counties.png",
    }

    for caption, path in maps.items():
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"Chart not found: {path}")

# ══════════════════════════════════════════════
# TAB 2 — PREDICTIVE TOOL
# ══════════════════════════════════════════════
with UserInterfaceTab:
    st.header("Loan Approval Prediction & Fairness Analysis")
    st.markdown(
        "Fill in your application details below. The tool runs two models — "
        "a **main model** trained on historical data and a **fair model** trained on "
        "financial factors only. The difference between them reveals the racial bias "
        "embedded in historical lending decisions."
    )

    st.divider()

    # ── INPUT FORM ────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Applicant Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            race = st.selectbox("Race / Ethnicity", [
                "White",
                "Black or African American",
                "Asian",
                "American Indian or Alaska Native",
                "Native Hawaiian or Other Pacific Islander",
                "Race Not Available",
            ])
            age = st.selectbox("Applicant Age", [
                "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "<25"
            ], index=1)
            has_co = st.radio("Co-Applicant?", ["No", "Yes"]) == "Yes"

        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=10000, max_value=2000000,
                                          value=250000, step=5000)
            dti = st.selectbox("Debt-to-Income Ratio", [
                "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
                "40%-<45%", "45%-<50%", "50%-60%", ">60%"
            ], index=2)
            ltv = st.slider("Loan-to-Value Ratio (LTV %)", 50, 100, 80)

        with col3:
            loan_purpose = st.selectbox("Loan Purpose", [
                "Home Purchase", "Home Improvement", "Refinancing",
                "Cash-out Refinancing", "Other"
            ])
            loan_term_years = st.slider("Loan Term", min_value=1, max_value=30, value=30, step=1, format="%d yrs")
            loan_term = loan_term_years * 12
            dwelling = st.selectbox("Property Type", [
                "Single Family (1-4 Units):Site-Built",
                "Single Family (1-4 Units):Manufactured",
                "Multifamily:Site-Built",
            ])

        st.subheader("Financial Context")
        col4, col5 = st.columns(2)
        with col4:
            area_income = st.number_input("Area Median Family Income ($)",
                                          min_value=20000, max_value=300000,
                                          value=78000, step=1000)
            credit_score_type = st.selectbox("Credit Score Model", [
                "Equifax Beacon 5.0",
                "Experian Fair Isaac",
                "FICO Risk Score Classic 04",
                "FICO Risk Score Classic 98",
                "Other",
            ])
        with col5:
            tract_pct = st.slider("Tract Income as % of Area Median", 50, 200, 100)

        submitted = st.form_submit_button("Predict Approval", type="primary", use_container_width=True)

    # ── RESULTS ──────────────────────────────
    if submitted:
        loan_purpose_map = {
            "Home Purchase": 1, "Home Improvement": 2, "Refinancing": 31,
            "Cash-out Refinancing": 32, "Other": 5
        }
        credit_map = {
            "Equifax Beacon 5.0": 1, "Experian Fair Isaac": 2,
            "FICO Risk Score Classic 04": 3, "FICO Risk Score Classic 98": 4, "Other": 9
        }

        inputs = {
            "race"             : race,
            "age"              : age,
            "has_co_applicant" : has_co,
            "loan_amount"      : loan_amount,
            "dti"              : dti,
            "ltv"              : ltv,
            "loan_purpose"     : loan_purpose_map[loan_purpose],
            "loan_term"        : loan_term,
            "dwelling"         : dwelling,
            "area_income"      : area_income,
            "credit_score_type": credit_map[credit_score_type],
            "tract_pct"        : tract_pct,
        }

        rf_prob, rf_decision, fair_prob, fair_decision, bias_gap = predict(inputs)

        st.divider()
        st.subheader("Results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                label="Main Model (Historical)",
                value=f"{rf_prob*100:.1f}%",
                delta="APPROVED" if rf_decision else "DENIED",
                delta_color="normal" if rf_decision else "inverse",
            )
            st.caption("Trained on historical HMDA data — includes racial signal")

        with col_b:
            st.metric(
                label="Fair Model (Finances Only)",
                value=f"{fair_prob*100:.1f}%",
                delta="APPROVED" if fair_decision else "DENIED",
                delta_color="normal" if fair_decision else "inverse",
            )
            st.caption("Trained on financial features only — no race, no demographics")

        with col_c:
            gap_label = f"{bias_gap:+.1f}%"
            st.metric(
                label="Bias Gap (Fair − Main)",
                value=gap_label,
                delta="No significant bias detected" if abs(bias_gap) < 3
                      else ("Racial penalty detected" if bias_gap < 0
                            else "Favorable adjustment"),
                delta_color="normal" if bias_gap >= -3 else "inverse",
            )
            st.caption("Negative gap = penalized beyond financial risk")

        # ── BIAS EXPLANATION ─────────────────
        st.divider()
        if abs(bias_gap) < 3:
            st.success(
                "The two models agree closely. Your predicted outcome appears "
                "consistent with your financial profile — no significant racial "
                "penalty detected."
            )
        elif bias_gap < 0:
            st.warning(
                f"**Racial Penalty Detected: {bias_gap:.1f}%**\n\n"
                f"Based on your finances alone, the fair model gives you "
                f"**{fair_prob*100:.1f}%** approval probability. "
                f"The historical model gives **{rf_prob*100:.1f}%**. "
                f"The **{abs(bias_gap):.1f}% gap** reflects racial bias "
                f"embedded in historical lending data — your finances justify "
                f"a higher approval chance than the historical model predicts."
            )
        else:
            st.info(
                f"The historical model is slightly more favorable than the fair model "
                f"for your profile ({bias_gap:+.1f}%). This can occur in profiles "
                f"where historical data showed above-average approval for your demographic group."
            )

        # ── LENDER RECOMMENDATIONS ───────────
        if race in MINORITY_RACES:
            st.divider()
            st.subheader("🏦 Recommended Lenders")
            st.markdown(
                f"These financial institutions in NC have historically shown the **smallest "
                f"approval gap** between White and **{race}** applicants with similar profiles. "
                f"Choosing a fairer lender can make a real difference."
            )

            recs = get_lender_recommendations(race)

            if recs is not None and len(recs) > 0:
                recs.columns = [c.replace("_", " ").title() for c in recs.columns]
                recs.index   = recs.index + 1
                st.dataframe(recs, use_container_width=True)
                st.caption(
                    "Gap = White approval rate − Minority approval rate. "
                    "Closer to 0 = fairer lender. Based on NC HMDA data 2019–2024."
                )
            else:
                st.info(
                    "Run `python lei_analysis.py` first to generate lender "
                    "fairness rankings."
                )

# ══════════════════════════════════════════════
# TAB 3 — RESEARCH CHARTS
# ══════════════════════════════════════════════
with ChartsTab:
    st.header("What the Research Showed Us")

    st.image(
        "charts/nc_county_loans_rate_scatter.png",
        caption="Loan amounts vs interest rates by county — shows how geography affects lending.",
        use_container_width=True,
    )
    st.image(
        "charts/chart3_rate_by_income_dti_race.png",
        caption="Interest rate by race — controlled for both income and DTI. Gaps here cannot be explained by financial risk alone.",
        use_container_width=True,
    )
    st.image(
        "charts/chart1_rate_by_dti_and_race.png",
        caption="Interest rate by race — controlled for DTI. Within each DTI bracket, racial gaps persist.",
        use_container_width=True,
    )
    st.image(
        "charts/chart2_rate_by_income_and_race.png",
        caption="Interest rate by race — controlled for income. Within each income bracket, racial gaps persist.",
        use_container_width=True,
    )