import pandas as pd
import numpy as np

mortgage_data = pd.read_csv("data_hmda_nc/nc_2019_2024.csv")



# drop rows where column is 1 or 1111
mortgage_data = mortgage_data[
    ~mortgage_data["business_or_commercial_purpose"].isin([1, 1111])
]

# optional: reset the index after dropping rows
mortgage_data.reset_index(drop=True, inplace=True)
#dropped 166,760 rows

# drop rows where interest_rate is NaN
mortgage_data = mortgage_data.dropna(subset=["interest_rate"])

# optional: reset index
mortgage_data.reset_index(drop=True, inplace=True)
#dropped 1,250,608 rows

# drop rows with any NaN values in any column
mortgage_data = mortgage_data.dropna()

# optional: reset the index
mortgage_data.reset_index(drop=True, inplace=True)


# drop rows with unwanted derived_race values
mortgage_data = mortgage_data[
    ~mortgage_data["derived_race"].isin(["Race Not Available", "Free Form Text Only"])
]

# optional: reset index
mortgage_data.reset_index(drop=True, inplace=True)

# drop rows with unwanted derived_ethnicity values
mortgage_data = mortgage_data[
    ~mortgage_data["derived_ethnicity"].isin(["Ethnicity Not Available", "Free Form Text Only"])
]

# optional: reset index
mortgage_data.reset_index(drop=True, inplace=True)
 

# drop rows with "Sex Not Available"
mortgage_data = mortgage_data[
    mortgage_data["derived_sex"] != "Sex Not Available"
]

# optional: reset index
mortgage_data.reset_index(drop=True, inplace=True)

# Filter out invalid placeholder
mortgage_data = mortgage_data[mortgage_data["applicant_age"] != "8888"]

