import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('churn_clean.csv')

#Univariate Continuous
Tenure = df['Tenure']
Outages = df['Outage_sec_perweek']

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.suptitle('Univariate Continuous Distributions')
plt.ylabel("Number of Customers")


ax1.hist(Tenure, color='lightgreen', ec='black', bins=50)
ax1.set_title('Tenure (bimodal distribution)')
ax1.set_xlabel("Months as a Customer")
ax2.hist(Outages, color='lightgreen', ec='black', bins=50)
ax2.set_title('Outages (normal distribution)')
ax2.set_xlabel("Average seconds per week without service")


#Univariate Categorical
fig, (ax3) = plt.subplots(1, sharey=True)
fig.suptitle('Internet Service (Univariate Categorical)')
plt.ylabel("Number of Customers")
plt.xlabel("Type of Internet")
# ax3 = df['Contract'].value_counts().plot.bar(rot=0)
ax3 = df['InternetService'].value_counts().plot.bar(rot=0)
plt.show()


# Bivariate Continuous
df.plot(x = 'Tenure', y = 'MonthlyCharge', kind = 'scatter')
plt.title("Monthly Cost based on Tenure (Non-Linear bivariate distribution)")
plt.show() 
df.plot(x = 'Tenure', y = 'Bandwidth_GB_Year', kind = 'scatter')
plt.title("Usage based on Tenure (Linear bivariate distribution)")
plt.show()  

# Bivariate Categorical
run_sum = np.array([0, 0, 0, 0, 0, 0, 0, 0])
x = ['1', '2', '3', '4', '5', '6', '7', '8']
colors = {'Item1': 'r', 'Item2': 'g', 'Item3': 'y', 'Item4': 'b', 'Item5': 'c', 'Item6': 'm', 'Item7': '0.5', 'Item8': 'k'}
for ind in ['Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8']:
    arr = np.array(df[[ind]].value_counts().sort_index())
    if len(arr) < 8:
        arr = np.append(arr, 0)
    
    plt.bar(x, arr, bottom=run_sum, color=colors[ind])
    run_sum += arr

plt.suptitle('Survey Responses (Bivariate Categorical)')
plt.ylabel("Number of Customers")
plt.xlabel("Survey Response (most to least important)")
plt.legend(['Timely response', 'Timely fixes', 'Timely replacements', 'Reliability', 'Options', 'Respectful response', 'Courteous exchange', 'Evidence of active listening'])
plt.show()


x = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
yes = np.array([])
no = np.array([])

for ind in x:
    arr = np.array(df[[ind]].value_counts().sort_index())
    yes = np.append(yes, arr[0])
    no = np.append(no, arr[1])

plt.bar(x, yes, bottom=no, color='b')
plt.bar(x, no, color='g')

plt.suptitle('Add Ons Distribution (Bivariate Categorical)')
plt.ylabel("Number of Customers")
plt.xlabel("Add Ons")
plt.legend(['Yes', 'No'])
plt.show()

