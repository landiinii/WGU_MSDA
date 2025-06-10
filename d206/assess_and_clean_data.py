import csv
import numpy as np
import pandas as pd
from scipy import stats
import math

# Collects metrics on the entire table and reports those metrics as specified in parameters passed
def show_metrics(df, all=False):    
    churn_variables = []
    churn_dtypes = []
    churn_count = []
    churn_unique = []
    churn_missing = []
    churn_pc_missing = []
    churn_std_dev = []
    max_num = []
    min_num = []
    max_len = []
    min_len = []
    
    for item in df.columns:
        if df[item].isna().sum() > 0 or all:
            churn_variables.append(item)
            churn_dtypes.append(df[item].dtype)
            churn_count.append(len(df[item]))
            churn_unique.append(len(df[item].unique()))
            churn_missing.append(df[item].isna().sum())
            churn_pc_missing.append(round((df[item].isna().sum() / len(df[item])) * 100, 2))
            if df[item].dtype == 'float64' or  df[item].dtype == 'int64':
                churn_std_dev.append(len(df[(np.abs(stats.zscore(df[item])) > 3)]))
                max_num.append(df[item].max())
                min_num.append(df[item].min())
                max_len.append(0)
                min_len.append(0)
            elif df[item].dtype == 'str':
                churn_std_dev.append(0)
                max_num.append(0)
                min_num.append(0)
                max_len.append(df[item].str.len().max())
                min_len.append(df[item].str.len().min())
            else:
                churn_std_dev.append(0)
                max_num.append(0)
                min_num.append(0)
                max_len.append(0)
                min_len.append(0)


    output = pd.DataFrame({
        'variable': churn_variables, 
        'dtype': churn_dtypes,
        'count': churn_count,
        'unique': churn_unique,
        'missing': churn_missing, 
        'pc_missing': churn_pc_missing,
        'standard_dev': churn_std_dev,
        'max_num': max_num,
        'min_num': min_num,
        'max_len': max_len,
        'min_len': min_len
    })    
        
    return output



def CorrectNulls(df):
    #correct for null values
    df["Bandwidth_GB_Year"].fillna(0, inplace=True)
    df["Phone"].fillna('No', inplace=True)
    df["TechSupport"].fillna('No', inplace=True)
    df["Children"].fillna(0, inplace=True)
    df["Techie"].fillna('No', inplace=True)
    df["InternetService"].fillna('None', inplace=True)
    df["Income"].fillna(df["Income"].mean(), inplace=True)
    df["Tenure"].fillna(df["Tenure"].mean(), inplace=True)
    df["Age"].fillna(round(df["Age"].mean()), inplace=True)
    print("Step 1: ", len(df))

    return df

def CorrectDataTypes(df):
    # correct and cast to boolean data types
    num_demo = ['Techie', 'Churn', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for c in num_demo:
        df[c] = df[c].map({'Yes': True, 'No': False})  # Replace string by boolean
    print("Step 2: ", len(df))

    return df

def CorrectOutliers(df):
    # correct numerics acros a standard deviation
    num_demo = ['Population', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'MonthlyCharge']
    std_dev = 3
    for c in num_demo:
        df = df[(np.abs(stats.zscore(df[c])) < std_dev)]
    print("Step 3: ", len(df))

    return df

def CorrectNumerals(df):
    # correct numerics within a range
    manual_demo = ['Population', 'Outage_sec_perweek']
    for c in manual_demo:
        df = df.drop(df[df[c] <= 0].index)
    print("Step 4: ", len(df))

    return df

def CorrectStrings(df):
    # ensure enumered variables are maintained valid
    enum_demo = {"Area": ['Rural', 'Urban', 'Suburban'], "Employment": ['Part Time', 'Retired', 'Student', 'Full Time', 'Unemployed'], "Marital": ['Married', 'Widowed', 'Divorced', 'Never Married', 'Separated'], "Gender": ['Male', 'Female', 'Prefer not to answer']}
    for c in enum_demo.keys():
        df = df[df[c].isin(enum_demo[c])]
    print("Step 5: ", len(df))

    # infinite strings = [City, State, County, Zip, Timezone, Job, Education, Churn]

    return df



df = pd.read_csv('churn_raw_data.csv')
asses = show_metrics(df, all=True)
asses.to_csv('assessment.csv', index=False)


df = pd.read_csv('churn_raw_data.csv', dtype={'Zip': 'str'})
df = CorrectNulls(df) # Step 1 of cleaning
df = CorrectDataTypes(df) # Step 2 of cleaning
df = CorrectOutliers(df) # Step 3 of cleaning
df = CorrectNumerals(df) # Step 4 of cleaning
df = CorrectStrings(df) # Step 5 of cleaning
print(show_metrics(df, all=True))
df.to_csv('churn_final_data.csv', index=False)
