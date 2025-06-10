import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


### CLEANING FUNCTIONS ###
def CorrectNulls(df):
    #correct for null values
    df.dropna(subset=['Churn'])
    df["Techie"].fillna('No', inplace=True)
    df["Outage_sec_perweek"].fillna(df["Outage_sec_perweek"].mean(), inplace=True)
    df["Email"].fillna(df["Email"].mean(), inplace=True)
    df["Contacts"].fillna(df["Contacts"].mean(), inplace=True)
    df["Yearly_equip_failure"].fillna(df["Yearly_equip_failure"].mean(), inplace=True)
    df["Tenure"].fillna(df["Tenure"].mean(), inplace=True)

    return df

def CorrectNumericOutliers(df):
    # correct numerics acros a standard deviation
    for c in ['Outage_sec_perweek','Email','Contacts','Yearly_equip_failure','Tenure']:
        df = df[(np.abs(stats.zscore(df[c])) < 3)]

    return df

def CorrectCategoricalOutliers(df):
    # ensure enumered variables are maintained valid
    enum_demo = {
        'Churn': ['Yes', 'No'], 
        'Techie': ['Yes', 'No']
        }
    for c in enum_demo.keys():
        df = df[df[c].isin(enum_demo[c])]

    return df

### TRANSFORMING FUNCTIONS ###

def NormalizeNumerics(df):
    for col in ['Children','Age','Income','Outage_sec_perweek','Email','Contacts','Yearly_equip_failure']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df

def OneHotEncodeCategoricals(df):
    df = pd.get_dummies(df, columns=['Marital'], drop_first=True)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    return df

def ConvertOrdinals(df):
    df['Area'] = df['Area'].map({'Rural': 0, 'Suburban': 0.5, 'Urban': 1}) # Replace string by float

    return df

def ConvertBinarys(df):
    df['Techie'] = df['Techie'].map({'Yes': 1, 'No': 0})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def ConvertDummys(df, columns):
    for c in df.columns.difference(columns):
        df[c] = df[c].map({True: 1, False: 0})
    
    return df



# Cleaning
columns = ['Churn','Tenure','Techie','Outage_sec_perweek','Email','Contacts','Yearly_equip_failure']
df = pd.read_csv('../churn_clean.csv')[columns]
print("Start: ", len(df))
df = CorrectNulls(df)
df = CorrectNumericOutliers(df)
df = CorrectCategoricalOutliers(df)
df.drop_duplicates(inplace=True)
print("End: ", len(df))

# Transforming
#df = NormalizeNumerics(df)
#df = OneHotEncodeCategoricals(df)
#df = ConvertOrdinals(df)
df = ConvertBinarys(df)
#df = ConvertDummys(df, columns)

# output
print(df.head())
df.to_csv('transformed_data.csv', index=False)

print(len(df.loc[df['Churn'] == 1]), len(df.loc[df['Churn'] == 0]))


train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)