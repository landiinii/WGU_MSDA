import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('transformed_data.csv')

label = df['MonthlyCharge']
features = df[df.columns.difference(['MonthlyCharge'])].assign(intercept = 1)

model = sm.OLS(label, features, hasconst=True)
results = model.fit()
print(results.summary())

reduced_columns = df.columns.difference(['MonthlyCharge', 'Contract', 
                                             'PaymentMethod_Mailed Check', 
                                             'PaymentMethod_Electronic Check', 
                                             'PaperlessBilling', 'Tablet', 
                                             'PaymentMethod_Credit Card (automatic)',
                                             'Port_modem', 'Phone', 'OnlineSecurity', 'InternetService_nan'])
refined_features = df[reduced_columns].assign(intercept = 1)

refined_model = sm.OLS(label, refined_features, hasconst=True)
refined_results = refined_model.fit()
print(refined_results.summary())

print("Residual Standard Error (initial MLR model): ")
print(results.resid.std(ddof=np.array(features).shape[1]))
print("Residual Standard Error (refined MLR model): ")
print(refined_results.resid.std(ddof=np.array(refined_features).shape[1]))


# residual plot
for col in reduced_columns:
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(refined_results, col, fig=fig)
    plt.show()
