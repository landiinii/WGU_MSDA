import numpy as np
import pandas as pd
from scipy import stats


def CorrectNulls(df):
    #correct for null values
    df["Port_modem"].fillna('No', inplace=True)
    df["Tablet"].fillna('No', inplace=True)
    df["InternetService"].fillna('None', inplace=True)
    df["Phone"].fillna('No', inplace=True)
    df["Multiple"].fillna('No', inplace=True)
    df["OnlineSecurity"].fillna('No', inplace=True)
    df["OnlineBackup"].fillna('No', inplace=True)
    df["DeviceProtection"].fillna('No', inplace=True)
    df["TechSupport"].fillna('No', inplace=True)
    df["StreamingTV"].fillna('No', inplace=True)
    df["StreamingMovies"].fillna('No', inplace=True)
    df["PaperlessBilling"].fillna('No', inplace=True)
    df["Tenure"].fillna(df["Tenure"].mean(), inplace=True)
    df["MonthlyCharge"].fillna(df["MonthlyCharge"].mean(), inplace=True)
    df["Bandwidth_GB_Year"].fillna(0, inplace=True)

    return df

def CorrectDichotomousTypes(df):
    # correct and cast to boolean data types
    num_demo = ['Port_modem', 'Tablet', 'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for c in num_demo:
        df = df[df[c].isin(['Yes', 'No'])] # remove rows without Yes or No values
        df[c] = df[c].map({'Yes': True, 'No': False}) # Replace string by boolean

    return df

def CorrectContinuousOutliers(df):
    # correct numerics acros a standard deviation
    num_demo = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']
    std_dev = 3
    for c in num_demo:
        df = df[(np.abs(stats.zscore(df[c])) < std_dev)]

    return df

def CorrectCategoricalOutliers(df):
    # ensure enumered variables are maintained valid
    enum_demo = {
        'Contract': ['Month-to-month', 'One year', 'Two Year'], 
        'InternetService': ['DSL', 'Fiber Optic', 'None'], 
        'PaymentMethod': ['Electronic Check', 'Mailed Check', 'Bank Transfer(automatic)', 'Credit Card (automatic)']}
    for c in enum_demo.keys():
        df = df[df[c].isin(enum_demo[c])]

    return df


# Step 1 of cleaning
df = pd.read_csv('../churn_clean.csv')
# Step 2 of cleaning
df = df[['Contract','Port_modem','Tablet','InternetService','Phone','Multiple','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod','Tenure','MonthlyCharge','Bandwidth_GB_Year']]
print("Start: ", len(df))
# Step 3 of cleaning
df = CorrectNulls(df)
# Step 4 of cleaning
df = CorrectDichotomousTypes(df)
# Step 5 of cleaning
df = CorrectContinuousOutliers(df)
# Step 6 of cleaning
df = CorrectCategoricalOutliers(df)
print("End: ", len(df))
df.to_csv('clean_data.csv', index=False)
