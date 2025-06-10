import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv('churn_clean.csv')

group1 = df[df['Churn']=='Yes']['Tenure']
group2 = df[df['Churn']=='No']['Tenure']
group3 = df['Tenure']

# perform independent two sample t-test
print(ttest_ind(group1, group2))

# perform Welch's t-test
print(ttest_ind(group1, group2, equal_var=False))


# Plot Distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
fig.suptitle('Churn Distributions')
plt.ylabel("Number of Customers")
plt.xlabel("Months as a Customer")

ax1.hist(group1, color='lightgreen', ec='black', bins=50)
ax1.set_title('Churned')

ax2.hist(group3, color='green', ec='black', bins=50)
ax2.set_title('Total')

ax3.hist(group2, color='darkgreen', ec='black', bins=50)
ax3.set_title('Current')

plt.show()
