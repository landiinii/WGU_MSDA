import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import lognorm

df = pd.read_csv('clean_data.csv')

# Univariate Dependent Distribution
plt.hist(df['MonthlyCharge'], color='lightgreen', ec='black', bins=50)
plt.title("Dependent Variable: Monthly Charge Distibution")
plt.xlabel("Average Monthly Bill in Dollars")
plt.show()
print()

#Univariate Continuous
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.suptitle('Independent Variables: Univariate Continuous Distributions')
plt.ylabel("Number of Customers")

ax1.hist(df['Tenure'], color='b', ec='black', bins=50)
ax1.set_title('Tenure')
ax2.hist(df['Bandwidth_GB_Year'], color='g', ec='black', bins=50)
ax2.set_title('Bandwidth_GB_Year')
plt.show()
print()


#Univariate Categorical
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 15))
fig.tight_layout()
df['Port_modem'].value_counts().plot(ax=axes[0, 0], kind='bar', rot=0, title = 'Port_modem')
df['Tablet'].value_counts().plot(ax=axes[0, 1], kind='bar', rot=0, title = 'Tablet');
df['Phone'].value_counts().plot(ax=axes[1, 0], kind='bar', rot=0, title = 'Phone')
df['Multiple'].value_counts().plot(ax=axes[1, 1], kind='bar', rot=0, title = 'Multiple')
df['OnlineSecurity'].value_counts().plot(ax=axes[2, 0], kind='bar', rot=0, title = 'OnlineSecurity')
df['OnlineBackup'].value_counts().plot(ax=axes[2, 1], kind='bar', rot=0, title = 'OnlineBackup')
df['DeviceProtection'].value_counts().plot(ax=axes[3, 0], kind='bar', rot=0, title = 'DeviceProtection')
df['TechSupport'].value_counts().plot(ax=axes[3, 1], kind='bar', rot=0, title = 'TechSupport')
df['StreamingTV'].value_counts().plot(ax=axes[4, 0], kind='bar', rot=0, title = 'StreamingTV')
df['StreamingMovies'].value_counts().plot(ax=axes[4, 1], kind='bar', rot=0, title = 'StreamingMovies')
df['PaperlessBilling'].value_counts().plot(ax=axes[5, 0], kind='bar', rot=0, title = 'PaperlessBilling')
df['Contract'].value_counts().plot(ax=axes[5, 1], kind='bar', rot=0, title = 'Contract')
df['InternetService'].value_counts().plot(ax=axes[6, 0], kind='bar', rot=0, title = 'InternetService')
df['PaymentMethod'].value_counts().plot(ax=axes[6, 1], kind='bar', rot=0, title = 'PaymentMethod')
plt.show()
print()




# Bivariate Continuous
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
fig.suptitle('Independent Variables: Bivariate Continuous Distributions')
plt.ylabel('Monthly Charge')

ax1.scatter(df['Tenure'], df['MonthlyCharge'])
ax1.set_title('Tenure x MonthlyCharge')
ax2.scatter(df['Bandwidth_GB_Year'], df['MonthlyCharge'])
ax2.set_title('Bandwidth x MonthlyCharge')
plt.show()
print()

# Bivariate Categorical
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20), sharey=True)
fig.tight_layout()

plt.ylabel('Monthly Charge')

columns = ['Port_modem','Tablet','Phone','Multiple','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
            'StreamingMovies','PaperlessBilling','Contract','InternetService','PaymentMethod']

for ind in range(len(columns)):
  row = math.floor(ind/3)
  col = ind%3
  axes[row, col].scatter(df[columns[ind]].astype(str), df['MonthlyCharge'])
  axes[row, col].set_title(columns[ind] + ' x MonthlyCharge')
plt.show()
print()