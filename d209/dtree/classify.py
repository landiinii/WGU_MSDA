import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier


def dtree(train, test):
    feature_cols = train.columns.difference(["Churn"])
    y_train = train["Churn"]
    X_train = train[feature_cols]
    y_test = test["Churn"]
    X_test = test[feature_cols]

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    plt.figure(figsize=(35, 30))
    tree.plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=["0", "1"],
        filled=True,
    )
    plt.show()


train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")
print(sum(test["Churn"]), len(test))
dtree(train, test)
