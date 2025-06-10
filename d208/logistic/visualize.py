import math

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm

df = pd.read_csv('clean_data.csv')

# Univariate Dependent Distribution
df['Churn'].value_counts().plot(kind='bar', rot=0, title = 'Churn')
plt.title("Dependent Variable: Churn Proportions")
plt.ylabel("Count of Customers")
plt.show()
print()

#Univariate Continuous
graph_col = 3
graph_rows = 4
fig, axes = plt.subplots(nrows=graph_rows, ncols=graph_col, figsize=(10, 10), sharey=True)
fig.tight_layout(pad=5.0)
fig.suptitle('Independent Variables: Univariate Continuous Distributions')
plt.ylabel("Number of Customers")

columns = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Population', 'Age', 'Children', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure']

for ind in range(len(columns)):
    row = math.floor(ind/graph_col)
    col = ind%graph_col

    axes[row, col].hist(df[columns[ind]], color='b', ec='black', bins=50)
    axes[row, col].set_title(columns[ind])

plt.show()
print()



#Univariate Categorical
graph_col = 3
graph_rows = 3
fig, axes = plt.subplots(nrows=graph_rows, ncols=graph_col, figsize=(10, 10), sharey=True)
fig.tight_layout(pad=5.0)
plt.ylabel('Count of Customers')


columns = ['Multiple', 'StreamingTV', 'StreamingMovies', 'Contract', 'InternetService', 'PaymentMethod', 'Area', 'Marital', 'Gender']

for ind in range(len(columns)):
    row = math.floor(ind/graph_col)
    col = ind%graph_col
    df[columns[ind]].value_counts().plot(ax=axes[row, col], kind='bar', title = columns[ind], xlabel='')

plt.show()
print()




# Bivariate Continuous
graph_col = 3
graph_rows = 4
fig, axes = plt.subplots(nrows=graph_rows, ncols=graph_col, figsize=(10, 10), sharey=True)
fig.tight_layout(pad=5.0)
fig.suptitle('Independent Variables: Univariate Continuous Distributions')

columns = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Population', 'Age', 'Children', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure']

for ind in range(len(columns)):
  row = math.floor(ind/graph_col)
  col = ind%graph_col
  norms = (df[columns[ind]] - df[columns[ind]].min()) / (df[columns[ind]].max() - df[columns[ind]].min()) 
  axes[row, col].scatter(df['Churn'].astype(str), norms)
  axes[row, col].set_title(columns[ind] + ' x Churn')
plt.show()
print()



#Bivariate Categorical
graph_col = 3
graph_rows = 3
fig, axes = plt.subplots(nrows=graph_rows, ncols=graph_col, figsize=(10, 10), sharey=True)
fig.tight_layout(pad=5.0)
plt.ylabel('Count of Customers')

columns = ['Multiple', 'StreamingTV', 'StreamingMovies', 'Contract', 'InternetService', 'PaymentMethod', 'Area', 'Marital', 'Gender']

for ind in range(len(columns)):
  row = math.floor(ind/3)
  col = ind%3

  df.groupby([columns[ind], 'Churn']).size().unstack().plot(ax=axes[row, col], kind='bar', stacked=True, title = columns[ind] + ' x MonthlyCharge', xlabel='')

plt.show()
print()