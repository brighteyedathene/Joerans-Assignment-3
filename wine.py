import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

wine_path = "data/winequality-white.csv"

def load_wine():
    dataset = pd.read_csv(wine_path, delimiter=';')
    headers = list(dataset.columns.values)
    import pdb; pdb.set_trace()

    return


load_wine()