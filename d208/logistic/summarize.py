import pandas as pd


def describeCategorical(df, col, dich=False):
  if dich:
    prop = df[col].value_counts(normalize=True)
    count = df[col].value_counts()
    return f"\nName: {col}\n" + f"Proportions: \tFalse: {prop[False]} \tTrue: {prop[True]}\n" + f"Count: \t\tFalse: {count[False]} \tTrue: {count[True]} \tTotal: {len(df[col])}\n"

  return "\nProportion:\n" + str(df[col].value_counts(normalize=True)) + "\nCount:\n" + str(df[col].value_counts())

df = pd.read_csv('clean_data.csv')

# Dependant Variable: Monthly Charge
print("Dependant Variable: ")
print(describeCategorical(df, 'Churn', dich=True))

# Independant Variables:
print("\n\nIndependant Continuous Variables: ")
print(df[['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Population']].describe())
print(df[['Age', 'Children', 'Income', 'Outage_sec_perweek']].describe())
print(df[['Email', 'Contacts', 'Yearly_equip_failure']].describe())


print("\n\nIndependant Dichotomous Variables: ")
for col in ['Multiple', 'StreamingTV', 'StreamingMovies']:
  print(describeCategorical(df, col, dich=True))
  


print("\n\nIndependant Categorical Variables: ")
for col in ['Contract', 'InternetService', 'PaymentMethod', 'Area', 'Marital', 'Gender']:
  print(describeCategorical(df, col))
