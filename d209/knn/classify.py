import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.neighbors import KNeighborsClassifier


def rocCurve(train, test):
    train_label = train['Churn']
    train_features = train[train.columns.difference(['Churn'])]
    model = KNeighborsClassifier(n_neighbors=9, weights='distance')
    model.fit(train_features, train_label)

    test_label = test['Churn']
    test_features = test[test.columns.difference(['Churn'])]
    y_scores = model.predict_proba(test_features)
    fpr, tpr, threshold = roc_curve(test_label, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    preds = model.predict(test_features)
    acc = accuracy_score(test_label, preds)
    print(acc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of kNN')
    plt.show()


def knn(train, test, k, weights):
    train_label = train['Churn']
    train_features = train[train.columns.difference(['Churn'])]
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    model.fit(train_features, train_label)

    test_label = test['Churn']
    test_features = test[test.columns.difference(['Churn'])]
    y_scores = model.predict_proba(test_features)
    fpr, tpr, threshold = roc_curve(test_label, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    

    # score = model.score(test_features, test_label)

    return roc_auc


def bestk(train, test):
    distance = []
    uniform = []
    for k in range(1, 10):
        distance.append(knn(train, test, k, 'distance'))
        uniform.append(knn(train, test, k, 'uniform'))
    average = np.average(np.subtract(np.array(distance), np.array(uniform)))
    print('Average: ' + str(average))
    print('Distance weight AUC: ' + str(distance))
    print('Uniform weight AUC: ' + str(uniform))




train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

# bestk(train, test)
rocCurve(train, test)




