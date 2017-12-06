import numpy as np
import pandas as pd

# might not need all this shit
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier


wine_path = "data/winequality-white.csv"

num_kfolds = 10

def load_wine():
    dataset = pd.read_csv(wine_path, delimiter=';')
    headers = list(dataset.columns.values)
    import pdb; pdb.set_trace()

    X = dataset.loc[:, headers[0:10]]
    y = dataset.loc[:, headers[11]]
    
    return X, y


classifiers = {
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Neural Network": MLPClassifier(alpha=1)
}

def main():
    X, y = load_wine()


    kfold = KFold(n_splits=num_kfolds)

    for name, clf in classifiers.iteritems():

        print name

        for train, test in kfold.split(X, y=y):
            clf.fit(X.iloc[train], y.iloc[train])
            score = clf.score(X.iloc[test], y[test])
            print score
            print '-------'

main()