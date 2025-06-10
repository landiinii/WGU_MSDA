import numpy as np
import pandas as pd
from scipy import stats
from dython.nominal import associations



def CorrectNulls(df):
    #correct for null values
    df["InternetService"].fillna('None', inplace=True)

    df["Churn"].fillna('No', inplace=True)
    df["StreamingTV"].fillna('No', inplace=True)
    df["StreamingMovies"].fillna('No', inplace=True)
    df["Multiple"].fillna('No', inplace=True)
    
    df["Tenure"].fillna(df["Tenure"].mean(), inplace=True)
    df["MonthlyCharge"].fillna(df["MonthlyCharge"].mean(), inplace=True)
    df["Bandwidth_GB_Year"].fillna(df["Bandwidth_GB_Year"].mean(), inplace=True)
    df["Population"].fillna(df["Population"].mean(), inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Children"].fillna(df["Children"].mean(), inplace=True)
    df["Income"].fillna(df["Income"].mean(), inplace=True)
    df["Outage_sec_perweek"].fillna(df["Outage_sec_perweek"].mean(), inplace=True)
    df["Email"].fillna(df["Email"].mean(), inplace=True)
    df["Contacts"].fillna(df["Contacts"].mean(), inplace=True)
    df["Yearly_equip_failure"].fillna(df["Yearly_equip_failure"].mean(), inplace=True)

    return df

def CorrectDichotomousTypes(df):
    # correct and cast to boolean data types
    num_demo = ['Churn', 'Multiple', 'StreamingTV', 'StreamingMovies']
    for c in num_demo:
        df = df[df[c].isin(['Yes', 'No'])] # remove rows without Yes or No values
        df[c] = df[c].map({'Yes': True, 'No': False}) # Replace string by boolean

    return df

def CorrectContinuousOutliers(df):
    # correct numerics acros a standard deviation
    num_demo = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Population', 'Age', 'Children', 
                'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure']
    std_dev = 3
    for c in num_demo:
        df = df[(np.abs(stats.zscore(df[c])) < std_dev)]

    return df

def CorrectCategoricalOutliers(df):
    # ensure enumered variables are maintained valid
    enum_demo = {
        'Contract': ['Month-to-month', 'One year', 'Two Year'], 
        'InternetService': ['DSL', 'Fiber Optic', 'None'], 
        'PaymentMethod': ['Electronic Check', 'Mailed Check', 'Bank Transfer(automatic)', 'Credit Card (automatic)'],
        'Area': ['Rural', 'Suburban', 'Urban'],
        'Marital': ['Widowed', 'Married', 'Separated', 'Never Married', 'Divorced'],
        'Gender': ['Male', 'Female', 'Nonbinary'],
        }
    for c in enum_demo.keys():
        df = df[df[c].isin(enum_demo[c])]

    return df




# Step 1 of cleaning
df = pd.read_csv('../churn_clean.csv')


# Step 2 of cleaning
# complete_correlation = associations(df, numerical_columns='auto', figsize=(20,20), cramers_v_bias_correction=False)
# df_complete_corr=complete_correlation['corr']
# print(df_complete_corr['Churn'])
# print(df.nunique())
df = df[['Churn','Population','Area','Children','Age','Income','Marital','Gender','Outage_sec_perweek','Email','Contacts','Yearly_equip_failure','Tenure','MonthlyCharge','Bandwidth_GB_Year','PaymentMethod', 'Contract', 'StreamingTV', 'StreamingMovies', 'Multiple', 'InternetService']]
print("Start: ", len(df))


# Step 3 of cleaning
df = CorrectNulls(df)

# Step 4 of cleaning
df = CorrectDichotomousTypes(df)

# Step 5 of cleaning
df = CorrectContinuousOutliers(df)

# Step 6 of cleaning
df = CorrectCategoricalOutliers(df)

# Step 7 of cleaning
df.drop_duplicates(inplace=True)

print("End: ", len(df))
df.to_csv('clean_data.csv', index=False)
