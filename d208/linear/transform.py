import pandas as pd

df = pd.read_csv('clean_data.csv')


print(df.head())

# Step 1 of transforming: Normalize the continuous variables to a 0-1 scale
for col in ['Tenure', 'Bandwidth_GB_Year']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) 

# Step 2 of transforming: cast boolean data types to integer representations of bits
num_demo = ['Port_modem', 'Tablet', 'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
for c in num_demo:
    df[c] = df[c].map({True: 1, False: 0}) # Replace boolean by int

# Step 3 of transforming: map contract terms to measures of months
df['Contract'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two Year': 24}) # Replace string by int

# Step 4 of transforming
# One Hot encode the remaining nominal variable using pandas native function
df = pd.get_dummies(df, columns=['InternetService'], drop_first=True, dummy_na=True)
df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)

newcols = df[df.columns.difference(num_demo + ['Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge', 'Contract'])].columns
for c in newcols:
    df[c] = df[c].map({True: 1, False: 0}) # Replace boolean by int

print(df.head())

df.to_csv('transformed_data.csv', index=False)
