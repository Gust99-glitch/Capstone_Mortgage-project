import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────
model = joblib.load("/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/models/model.joblib")
feature_names = joblib.load("/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/models/feature_names.joblib")
train_medians = joblib.load("/Users/gustave/Desktop/dtsc/Capstone_Mortgage-project/models/train_medians.joblib")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Enter borrower and loan details to predict approval probability.")

# ─────────────────────────────────────────────
# USER INPUTS
# ─────────────────────────────────────────────
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
income = st.number_input("Income", min_value=0.0, value=60000.0)
property_value = st.number_input("Property Value", min_value=0.0, value=250000.0)
loan_term = st.number_input("Loan Term (months)", min_value=0, value=360)
population = st.number_input("Tract Population", min_value=1, value=5000)

dti_options = [
    "<20%", "20%-<30%", "30%-<36%", "36%-<40%",
    "40%-<45%", "45%-<50%", "50%-60%", ">60%", "Exempt"
]
dti = st.selectbox("Debt-to-Income Ratio", dti_options)

# ─────────────────────────────────────────────
# ENCODE INPUT
# ─────────────────────────────────────────────
dti_map = {v: i for i, v in enumerate(dti_options)}

input_dict = {
    "loan_amount": loan_amount,
    "income": income,
    "property_value": property_value,
    "loan_term": loan_term,
    "tract_population": population,
    "debt_to_income_ratio": dti_map[dti],
}

# Feature engineering (same as training)
input_dict["loan_to_income"] = loan_amount / (income + 1)
input_dict["loan_to_value"] = loan_amount / (property_value + 1)
input_dict["income_per_person"] = income / (population + 1)

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Add missing columns (VERY IMPORTANT)
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_names]

# Fill missing values
input_df = input_df.fillna(pd.Series(train_medians))

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Approval Probability: {prob:.2%}")

    if prob >= 0.5:
        st.success("✅ Likely Approved")
    else:
        st.error("❌ Likely Denied")