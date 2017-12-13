import numpy as np
import pandas as pd

# metrics and misc
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import KFold

# regression
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.preprocessing import PolynomialFeatures

# classifiers
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

# this lets you get a dataframe instead of a numpy array
#from sklearn_pandas import DataFrameMapper


wine_path = "data/winequality-white.csv"

num_kfolds = 15

def load_wine():
    dataset = pd.read_csv(wine_path, delimiter=';')
    headers = list(dataset.columns.values)

    X = dataset.loc[:, headers[0:10]]
    y = dataset.loc[:, headers[11]]
    
    return X, y


classifiers = {
    #"Linear SVM": SVC(kernel="linear", C=0.25),
    #"RBF SVM": SVC(gamma=2, C=1),
    "Neural Network": MLPClassifier(alpha=0.0001, activation='relu'),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
    "Random Forest Classifier": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Ada Boost Classifier": AdaBoostClassifier(),
    "Quadratic DiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0)), ## took too long
    #"Gaussian Naive Bayes": GaussianNB() ## this one just sucked

}

regressors = {
    "Linear Regression": LinearRegression()
}

def main():
    X, y = load_wine()

    distributions = [
        ('Unscaled data', X),
        ('Data after standard scaling',
            StandardScaler().fit_transform(X)),
        ('Data after min-max scaling',
            MinMaxScaler().fit_transform(X)),
        ('Data after max-abs scaling',
            MaxAbsScaler().fit_transform(X)),
        ('Data after robust scaling',
            RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
        ('Data after quantile transformation (uniform pdf)',
            QuantileTransformer(output_distribution='uniform')
            .fit_transform(X)),
        ('Data after quantile transformation (gaussian pdf)',
            QuantileTransformer(output_distribution='normal')
            .fit_transform(X)),
        ('Data after sample-wise L2 normalizing',
            Normalizer().fit_transform(X))
    ]

    for label, scaled_X in distributions:
        print '========================'
        print label
        print '========================'
        scaled_X_df = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)
        machine_learn(scaled_X_df, y)
        print '========================'
        print ''

def machine_learn(X, y):
    kfold = KFold(n_splits=num_kfolds)

    #import pdb; pdb.set_trace()

    for name, clf in classifiers.iteritems():
        print name
        try:
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

            #print "soft accuracy:"
            #PlotSoftAccuracy(soft_accuracy_sum)
            normalized_soft_accuracy = NormalizeSoftAccuracy(soft_accuracy)
            #PlotSoftAccuracy(normalized_soft_accuracy)

            errors = CombineEquivalentErrorSoftAccuracy(normalized_soft_accuracy)
            PlotErrors(errors)
        except ValueError as e:
            print "didn't work: ", e 
    
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

def PlotErrors(errors):
    print 'Error'
    percentage_sum = 0
    for key, value in errors.iteritems():
        percentage_sum += value
        print '+/-', key, ':', value, '  (', percentage_sum, ')'

if __name__ == '__main__':
    main()