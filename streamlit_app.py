import streamlit as st
import pandas as pd
import numpy as np
import modal
import joblib
import pathlib
import os
# Loading basic model artifacts

# rf_model = joblib.load("models/rf_model.joblib")
# xgb_model = joblib.load("models/xgb_model.joblib")
# rf_features = joblib.load("models/rf_feature_names.joblib")
# xgb_features = joblib.load("models/xgb_feature_names.joblib")
# rf_medians = joblib.load("models/rf_train_medians.joblib")
# xgb_medians = joblib.load("models/xgb_train_medians.joblib")

# Title of Demo
st.title("Mortgage Markets in Motion - The Demo")

# Adding Tab Groups

HeatMapTab, UserInterfaceTab, ChartsTab = st.tabs(["📍 NC Heatmap", "💭 Predictive Tool", "📈 Research Charting"], default="💭 Predictive Tool")

# Creating 

with ChartsTab:
    st.header("What the Research Showed Us")
    st.image("charts/nc_county_loans_rate_scatter.png", caption="This chart was an interesting showcase of how rates and loan amounts correlate.")
    st.image("charts/chart3_rate_by_income_dti_race.png", caption="A combined chart of DTI, income groups, and racial demographics. The y-axis here is loan rates.")
    st.image("charts/chart1_rate_by_dti_and_race.png", caption="Closer look at DTI and racial demographics.")
    st.image("charts/chart2_rate_by_income_and_race.png", caption="Closer look at income groups and racial demographics.")

with HeatMapTab:
    st.header("A Geographic View of NC Loans")
    file_path_0 = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_3panel_county_maps.png"
    if file_path_0.exists():
        st.image(str(file_path_0), caption="A combined map of all three key loan stats for each county in NC.")
    else:
        st.error(f"File not found: {file_path_0}")

    heatmap_total = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_total_loans_map.png"   
    heatmap_avgL = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_avg_loan_map.png"
    heatmap_avgR = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_avg_rate_map.png"

    option = st.selectbox("Select a County Heatmap to View:", ["Total Loans", "Average Loans", "Average Rates"])
    
    if option == "Total Loans":
        st.image(str(heatmap_total), caption="Heatmap showing which counties have the most loans and which have the least.")
    elif option == "Average Loans":
        st.image(str(heatmap_avgL), caption="Heatmap displaying which counties have the highest and lowest average loan amounts.")
    elif option == "Average Rates":
        st.image(str(heatmap_avgR), caption="Heatmap showcasing which counties have the highest and lowest average rates.")
   
# OLD CODE BELOW ---------------------------
    # file_path_1 = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_total_loans_map.png"
    # if file_path_1.exists():
    #     st.image(str(file_path_1))
    # else:
    #     st.error(f"File not found: {file_path_1}")
    
    # file_path_2 = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_avg_loan_map.png"
    # if file_path_2.exists():
    #     st.image(str(file_path_2))
    # else:
    #     st.error(f"File not found: {file_path_2}")
    
    # file_path_3 = pathlib.Path(__file__).parent / r"C:\Users\valni\AppData\Local\Programs\DTSCProjects\Capstone\Capstone_Mortgage-project\charts\nc_avg_rate_map.png"
    # if file_path_3.exists():
    #     st.image(str(file_path_3))
    # else:
    #     st.error(f"File not found")

    # Extra stuff, might work might not streamlit and python do what they want    
    # st.image("charts/nc_total_loan_map.png", caption="Heatmap displaying highest and lowest total loans by county.")
    # st.image("charts/nc_avg_loan_map.png", caption="Heatmap showing which NC counties had the highest averag loans.")
    # st.image("charts/nc_avg_rate_map.png", caption="Heatmap displaying the highest and lowest rates for all counties in NC.")


with UserInterfaceTab:
    st.header("The Predictive Models and User Tools")
