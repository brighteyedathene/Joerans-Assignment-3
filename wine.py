import numpy as np
import pandas as pd

# might not need all this shit
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import KFold

from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier


wine_path = "data/winequality-white.csv"

num_kfolds = 10

def load_wine():
    dataset = pd.read_csv(wine_path, delimiter=';')
    headers = list(dataset.columns.values)

    X = dataset.loc[:, headers[0:10]]
    y = dataset.loc[:, headers[11]]
    
    return X, y


classifiers = {
    #"Linear SVM": SVC(kernel="linear", C=0.025),
    #"RBF SVM": SVC(gamma=2, C=1),
    "Neural Network": MLPClassifier(alpha=1)
}

regressors = {
    "Linear Regression": LinearRegression()
}

def main():
    X, y = load_wine()


    kfold = KFold(n_splits=num_kfolds)

    for name, clf in classifiers.iteritems():
        print name

        accuracy_sum = 0
        soft_accuracy_sum = {}
        for train, test in kfold.split(X, y=y):
            print '.',
            clf.fit(X.iloc[train], y.iloc[train])
            prediction = clf.predict(X.iloc[test])
            accuracy = accuracy_score(y[test], prediction)

            soft_accuracy = GetSoftAccuracy(prediction, y[test])

            soft_accuracy_sum = AddSoftAccuracy(soft_accuracy_sum, soft_accuracy)

            accuracy_sum += accuracy

        accuracy_average = accuracy_sum / num_kfolds
        print "\nAccuracy score:", accuracy_average

        print "soft accuracy:"
        PlotSoftAccuracy(soft_accuracy_sum)

    for name, reg in regressors.iteritems():
        print name

        r2_sum = 0
        for train, test in kfold.split(X, y=y):
            print '.',
            reg.fit(X.iloc[train], y.iloc[train])
            prediction = reg.predict(X.iloc[test])
            r2 = r2_score(y.iloc[test], prediction)

            r2_sum += r2

        r2_average = r2_sum / num_kfolds
        print '\nr2 score:', r2_average


def GetSoftAccuracy(prediction, actual):
    '''Map the difference between prediction and actual
    into indices from -10 to +10

    We count the occurences of each difference from -10 to +10
    ie. how many times the prediction was off by -1, 0, +1, +2 etc

    After, we return a dict{-10:<number of -10s>, -9:<number of -9s> ...}
    '''

    # Count occurences of each difference
    differences     = np.subtract(prediction, actual)
    uniques, counts = np.unique(differences, return_counts=True)
    soft_accuracy   = dict(zip(uniques, counts))

    labels = range(-10, 11)
    for label in labels:
        if not soft_accuracy.get(label):
            soft_accuracy[label] = 0

    return soft_accuracy

def AddSoftAccuracy(A, B):
    result = {}
    for index in range(-10, 11):
        result[index] = A.get(index, 0) + B.get(index, 0)

    return result

def NormalizeSoftAccuracy(raw_soft_accuracy):
    normalized_soft_accuracy = {}
    count_max = max(raw_soft_accuracy.itervalues())
    for key, value in raw_soft_accuracy.iteritems():
        normalized_soft_accuracy[key] = value / count_max

    return normalized_soft_accuracy


def PlotSoftAccuracy(soft_accuracy):
    '''accuracy_index should be a 21 element list 
    containing the counts of the differences bewteen
    prediction and actual
    '''
    for index in range(-10, 11):
        print index, ':', soft_accuracy.get(index, 0)


if __name__ == '__main__':
    main()