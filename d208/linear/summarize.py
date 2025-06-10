import pandas as pd


df = pd.read_csv('clean_data.csv')

# Dependant Variable: Monthly Charge
print("Dependant Variable: ")
print(df['MonthlyCharge'].describe())

# Independant Variables:
print("\n\nIndependant Continuous Variables: ")
print(df[['Tenure','Bandwidth_GB_Year']].describe())

print("\n\nIndependant Dichotomous Variables: ")
for col in ['Port_modem','Tablet','Phone','Multiple','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
            'StreamingMovies','PaperlessBilling']:
  prop = df[col].value_counts(normalize=True)
  count = df[col].value_counts()
  print(f"\nName: {col}")
  print(f"Proportions: \tFalse: {prop[False]} \tTrue: {prop[True]}")
  print(f"Count: \t\tFalse: {count[False]} \tTrue: {count[True]} \tTotal: {len(df[col])}")


print("\n\nIndependant Categorical Variables: ")
for col in ['Contract', 'InternetService','PaymentMethod']:
  print("\nProportion:")
  print(df[col].value_counts(normalize=True))
  print("Count:")
  print(df[col].value_counts())
