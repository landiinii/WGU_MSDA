import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics




def statsmodels(df):
    label = df['Churn']
    features = df[df.columns.difference(['Churn'])].assign(intercept = 1)

    model = sm.Logit(label, features)
    results = model.fit()
    print(results.summary2())

    reduced_columns = df.columns.difference(['Churn', 'Outage_sec_perweek', 'Yearly_equip_failure', 
                                            'Population', 'Income', 'Email', 'Area_Urban', 'Contacts',
                                            'Marital_Married', 'Marital_Separated', 'Gender_Nonbinary', 
                                            'Area_Suburban', 'Children', 'Marital_Never Married', 'Age', 
                                            'PaymentMethod_Credit Card (automatic)', 'PaymentMethod_Mailed Check'])
    refined_features = df[reduced_columns].assign(intercept=1)

    refined_model = sm.Logit(label, refined_features)
    refined_results = refined_model.fit()
    print(refined_results.summary2())

    overall_bic = math.inf
    added_columns = []

    while True:
        to_add_columns = df.columns.difference(['Churn'] + added_columns)
        running_bic = math.inf
        running_col = ''
        for i in to_add_columns:
            features = df[added_columns + [i]].assign(intercept=1)
            model = sm.Logit(label, features)
            results = model.fit(disp=False)
            if results.bic < running_bic:
                running_bic = results.bic
                running_col = i

        if overall_bic > running_bic:
            added_columns = added_columns + [running_col]
            overall_bic = running_bic
            print(running_col, running_bic)
        else:
            break

    print(added_columns)

    refined_features = df[added_columns].assign(intercept=1)

    refined_model = sm.Logit(label, refined_features)
    refined_results = refined_model.fit()
    print(refined_results.summary2())

def sklearn(df):
    label = df['Churn']
    reduced_columns = df.columns.difference(['Churn', 'Outage_sec_perweek', 'Yearly_equip_failure', 
                                            'Population', 'Income', 'Email', 'Area_Urban', 'Contacts',
                                            'Marital_Married', 'Marital_Separated', 'Gender_Nonbinary', 
                                            'Area_Suburban', 'Children', 'Marital_Never Married', 'Age', 
                                            'PaymentMethod_Credit Card (automatic)', 'PaymentMethod_Mailed Check'])
    refined_features = df[reduced_columns].assign(intercept=1)

    logisticRegr = LogisticRegression(max_iter=10000)

    scores = []
    for ind in range(10):
        start = math.ceil(len(refined_features)/10) * ind
        stop = start + math.ceil(len(refined_features)/10)
        x_train = pd.concat([refined_features[:start], refined_features[stop:]])
        x_test = refined_features[start:stop]
        y_train = pd.concat([label[:start], label[stop:]])
        y_test = label[start:stop]


        logisticRegr.fit(x_train, y_train)
        score = logisticRegr.score(x_test, y_test)
        scores.append(score)
    print("Accuracy: " + str(sum(scores) / len(scores)) )


    confusion_matrix = metrics.confusion_matrix(label, logisticRegr.predict(refined_features))
    print(confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()


df = pd.read_csv('transformed_data.csv')
statsmodels(df)
sklearn(df)
