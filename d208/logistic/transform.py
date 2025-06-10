import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('clean_data.csv')


print(df.head())

# Step 1 of transforming: Normalize the continuous variables to a 0-1 scale
columns = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Population', 'Age', 'Children', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure']
for col in columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) 

# Step 2 of transforming: cast boolean data types to integer representations of bits
num_demo = ['Multiple', 'StreamingTV', 'StreamingMovies']
for c in num_demo:
    df[c] = df[c].map({True: 1, False: 0}) # Replace boolean by int

# Step 3 of transforming: map contract terms to measures of months
df['Contract'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two Year': 24}) # Replace string by int

# Step 4 of transforming
# One Hot encode the remaining nominal variable using pandas native function
df = pd.get_dummies(df, columns=['InternetService'], drop_first=True, dummy_na=True)
df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)
df = pd.get_dummies(df, columns=['Area'], drop_first=True)
df = pd.get_dummies(df, columns=['Marital'], drop_first=True)
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

newcols = df[df.columns.difference(num_demo + columns + ['Churn', 'Contract'])].columns
for c in newcols:
    df[c] = df[c].map({True: 1, False: 0}) # Replace boolean by int

# Step 5 of transforming
# Run VIF Analysis to remove variables with high colinearilty
X = df[df.columns.difference(['Churn'])] # Churn removed since its the output
vif = pd.DataFrame() 
vif["feature"] = X.columns 
vif["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
print(vif)

df = df[df.columns.difference(['Bandwidth_GB_Year'])] # Bandwidth_GB_Year removed for colinearity

print(df.head())

df.to_csv('transformed_data.csv', index=False)
