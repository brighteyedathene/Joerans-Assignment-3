import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import make_plots
# regression
from sklearn.linear_model import LinearRegression
# metrics and misc
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
# classifiers
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
# transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier



wine_path = "data/winequality-white.csv"

num_kfolds = 10


def load_wine():
    dataset = pd.read_csv(wine_path, delimiter=';')
    headers = list(dataset.columns.values)

    X = dataset.loc[:, headers[0:10]]
    y = dataset.loc[:, headers[11]]

    return X, y


classifiers = {
    #"Linear SVM": SVC(kernel="linear", C=0.25),
    "RBF SVM: gamma=2, c=1": SVC(gamma=2, C=1),
    #"RBF SVM: gamma=2, c=0.1": SVC(gamma=2, C=0.1),
    #"RBF SVM: gamma=4, c=1": SVC(gamma=4, C=1),
    #"RBF SVM: gamma=4, c=0.1": SVC(gamma=4, C=0.1),
    "Neural Network: relu, alpha=0.0001": MLPClassifier(alpha=0.0001, activation='relu'),
    #"Neural Network: relu, alpha=0.01": MLPClassifier(alpha=0.01, activation='relu'),
    #"Neural Network: relu, alpha=1": MLPClassifier(alpha=1, activation='relu'),
    #"Neural Network: tanh": MLPClassifier(alpha=0.0001, activation='tanh'),
    #"Neural Network: logistic": MLPClassifier(alpha=0.0001, activation='logistic'),
    #"Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
    #"Random Forest Classifier": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #"Ada Boost Classifier": AdaBoostClassifier(),
    #"Quadratic DiscriminantAnalysis": QuadraticDiscriminantAnalysis(), ## doesn't work
    #"Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0)), ## took too long
    #"Gaussian Naive Bayes": GaussianNB() ## this one just sucked
}

scores = {}

regressors = {
    #"Linear Regression": LinearRegression()
}


def main():
    X, y = load_wine()

    distributions = [
        #('Unscaled data', X),
        #('Data after standard scaling', StandardScaler().fit_transform(X)),
        #('Data after min-max scaling', MinMaxScaler().fit_transform(X)),
        #('Data after max-abs scaling', MaxAbsScaler().fit_transform(X)),
        #('Data after robust scaling', RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
        ('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform').fit_transform(X)),
        #('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal').fit_transform(X)),
        #('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X)) ## this one sucked
    ]

    for label, scaled_X in distributions:
        print '========================'
        print label
        print '========================'
        scaled_X_df = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)
        machine_learn(scaled_X_df, y)
        print '========================'
        print ''

    make_plots(scores)


def machine_learn(X, y):
    kfold = KFold(n_splits=num_kfolds)

    # import pdb; pdb.set_trace()

    for name, clf in classifiers.iteritems():
        print name
        try:
            all_predictions = np.empty(0)
            prediction_max = 0
            prediction_min = 10
            accuracy_sum = 0
            accuracy_min = 1
            accuracy_max = 0
            soft_accuracy_sum = {}
            t0 = time.clock()
            for train, test in kfold.split(X, y=y):
                print '.',
                clf.fit(X.iloc[train], y.iloc[train])
                prediction = clf.predict(X.iloc[test])
                accuracy = accuracy_score(y[test], prediction)

                soft_accuracy = GetSoftAccuracy(prediction, y[test])
                soft_accuracy = NormalizeSoftAccuracy(soft_accuracy)
                soft_accuracy_sum = AddSoftAccuracy(soft_accuracy_sum, soft_accuracy)

                np.append(all_predictions, prediction)
                prediction_max = max(prediction_max, np.max(prediction))
                prediction_min = min(prediction_min, np.min(prediction))

                accuracy_min = min(accuracy_min, accuracy)
                accuracy_max = max(accuracy_max, accuracy)
                accuracy_sum += accuracy

            accuracy_average = accuracy_sum / num_kfolds
            time_taken = time.clock() - t0
            normalized_soft_accuracy = NormalizeSoftAccuracy(soft_accuracy_sum)
            errors = CombineEquivalentErrorSoftAccuracy(normalized_soft_accuracy)
            error_list = ErrorsAsList(errors)
            print "Completed %s k-folds" % num_kfolds
            print "Time taken: ", time_taken, " seconds"
            print "Accuracy score:", accuracy_average
            print "Accuracy Min:", accuracy_min
            print "Accuracy Max:", accuracy_max
            print "Prediction Max:", prediction_max
            print "Prediction Nax:", prediction_min
            PlotErrors(errors)
            scores[name] = errors

            plt.title("Prediction Distribution (" + name + ")")
            plt.xlabel("Predicted Rating")
            plt.ylabel("Count")
            plt.xlim([0,10])
            plt.hist(prediction, 10, range=(0,10), alpha=0.45)
            plt.show()

        except ValueError as e:
            print "didn't work: ", e



    for name, reg in regressors.iteritems():
        print name

        rmse_sum = 0
        for train, test in kfold.split(X, y=y):
            print '.',
            reg.fit(X.iloc[train], y.iloc[train])
            prediction = reg.predict(X.iloc[test])
            rmse = math.sqrt(mean_squared_error(y.iloc[test], prediction))

            rmse_sum += rmse

        rmse_average = rmse_sum / num_kfolds
        print '\nRoot mean error:', rmse_average


def GetSoftAccuracy(prediction, actual):
    '''
    Map the difference between prediction and actual
    into indices from -10 to +10

    We count the occurences of each difference from -10 to +10
    ie. how many times the prediction was off by -1, 0, +1, +2 etc

    After, we return a dict{-10:<number of -10s>, -9:<number of -9s> ...}
    '''

    # Count occurences of each difference
    differences = np.subtract(prediction, actual)
    uniques, counts = np.unique(differences, return_counts=True)
    soft_accuracy = dict(zip(uniques, counts))

    labels = range(-10, 11)
    for label in labels:
        if not soft_accuracy.get(label):
            soft_accuracy[label] = 0

    return soft_accuracy


def AddSoftAccuracy(A, B):
    result = {}
    for index in xrange(-10, 11):
        result[index] = A.get(index, 0) + B.get(index, 0)

    return result


def NormalizeSoftAccuracy(raw_soft_accuracy):
    normalized_soft_accuracy = {}
    count_max = sum(raw_soft_accuracy.itervalues())
    for key, value in raw_soft_accuracy.iteritems():
        normalized_soft_accuracy[key] = float(value) / count_max

    return normalized_soft_accuracy


def PlotSoftAccuracy(soft_accuracy):
    '''accuracy_index should be a 21 element list 
    containing the counts of the differences bewteen
    prediction and actual
    '''
    for index in xrange(-10, 11):
        print index, ':', soft_accuracy.get(index, 0)


def CombineEquivalentErrorSoftAccuracy(soft_accuracy):
    equivalent_errors = {}
    equivalent_errors[0] = soft_accuracy.get(0, 0)
    for x in xrange(1, 11):
        equivalent_errors[x] = soft_accuracy.get(-x, 0) + soft_accuracy.get(x, 0)

    return equivalent_errors

def ErrorsAsList(errors):
    error_list = []
    for v in errors.itervalues():
        error_list.append(v)

    return error_list

def PlotErrors(errors):
    print 'Error'
    percentage_sum = 0
    for key, value in errors.iteritems():
        percentage_sum += value
        print '+/-', key, ':', value, '  (', percentage_sum, ')'
        if percentage_sum >= 1.0:
            return


if __name__ == '__main__':
    main()
