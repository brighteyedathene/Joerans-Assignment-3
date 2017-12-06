import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# quality of life imports
from dateutil.parser import parse
from time import mktime, time
import datetime
import functools
import json

with open('config.json') as file:
    config = json.load(file)

movielens_directory = config['movielens_directory']
movielens_users = movielens_directory + 'u.user'
movielens_data = movielens_directory + 'u.data'
movielens_items = movielens_directory + 'u.item'


def load_movielens():
    print "Loading movielens..."
    # pass in column names for each CSV
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(movielens_users, sep='|', names=u_cols,
                        encoding='latin-1')

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(movielens_data, sep='\t', names=r_cols,
                          encoding='latin-1')

    m_cols = ['movie_id', 'title' , 'release_date' , 'video_release_date' ,
              'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
              "Children's", 'Comedy', 'Crime' , 'Documentary', 'Drama', 'Fantasy',
              'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
              'Thriller' , 'War' , 'Western']

    movies = pd.read_csv(movielens_items, sep='|', names=m_cols, encoding='latin-1')

    # create one merged DataFrame
    movie_ratings = pd.merge(movies, ratings)
    lens = pd.merge(movie_ratings, users)

    # Limit the dataset to something more manageable on this shit computer
    top30 = lens.groupby('title').size().sort_values(ascending=False)[:800].keys()
    print "Ignore that warning - Just a handy way to get the top N movie titles"
    lens = lens.loc[lens['title'].isin(top30)]

    # This vectorized function helps generate features from a list of text fields
    text_matcher = lambda item_name, column_name: 1 if column_name is item_name else 0
    vectorized_matcher = np.vectorize(text_matcher)

    # Vectorize movie_id ## Can't do this!
    t0 = time()
    print "Vectorizing movie titles (this might take a while)"
    titles = lens.loc[:, ['title']].drop_duplicates()
    titles_array = [title.item() for title in titles._values]
    print "Vectorizing %s titles" % len(titles_array)
    for title in titles_array:
        lens[title] = vectorized_matcher(lens.loc[:, 'title']._values, title)
    t1 = time()
    print "done in %s" % (t1 - t0)

    # Vectorize occupation
    occs = lens.loc[:, ['occupation']].drop_duplicates()
    occupation_strings = [occ.item() for occ in occs._values]
    for occupation in occupation_strings:
        lens[occupation] = vectorized_matcher(lens.loc[:, 'occupation']._values, occupation)

    # Vectorize sex    
    sexes = lens.loc[:, ['sex']].drop_duplicates()
    sex_strings = [sex.item() for sex in sexes._values]
    for sex in sex_strings:
        lens[sex] = vectorized_matcher(lens.loc[:, 'sex']._values, sex)

    # Convert dates to timestamps
    date_to_stamp = lambda date: mktime(parse(date).timetuple()) if date==date else 0
    vectorized_date_to_stamp = np.vectorize(date_to_stamp, otypes=[np.float])

    lens['release_date_stamp'] = vectorized_date_to_stamp(lens.loc[:, 'release_date']._values)

    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)

    genres = [
        'Action', 'Adventure', 'Animation',
        "Children's", 'Comedy', 'Crime' , 'Documentary', 'Drama', 'Fantasy',
        'Film_Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci_Fi' ,
        'Thriller' , 'War' , 'Western'
    ]
    occupations = [
        'retired', 'salesman', 'engineer', 'librarian', 'student', 'other', 
        'executive','administrator', 'artist', 'entertainment', 'educator', 
        'scientist', 'none', 'programmer', 'homemaker', 'marketing', 
        'technician', 'lawyer', 'writer', 'healthcare', 'doctor'
    ]

    features = ['age', 'M', 'F',                
                'release_date_stamp',
                'unix_timestamp'
                ] + titles_array# + genres + occupations
    
    target = ['rating']


    X = lens.loc[:, features]
    y = lens.loc[:, target]

    print "Working with %s rows" % len(X)

    return X, y


def kfold(prediction_function, X, y, row_count, k=10):

    # get indices for X and y
    X = X.loc[0:row_count]
    y = y.loc[0:row_count]

    kfold = KFold(n_splits=k)

    rmse_array = []
    r2_array = []
    for train, test in kfold.split(X, y=y):

        prediction = prediction_function(X, y, train, test)

        # RMSE Evalutation
        mse = mean_squared_error(y.iloc[test], prediction)
        rmse = np.sqrt(mse)
        rmse_array.append(rmse)

        # Coeffitient of Determination (R squared) Evaluation
        r2 = r2_score(y.iloc[test], prediction)
        r2_array.append(r2)

    average_rmse = sum(rmse_array) / k
    average_r2 = sum(r2_array) / k
    return average_rmse, average_r2


def predict_linear(X, y, train, test):
    # Fit
    linereg = LinearRegression()
    linereg.fit(X.iloc[train], y.iloc[train])

    # Predict
    prediction = linereg.predict(X.iloc[test])
    return prediction

def predict_ridge(X, y, train, test, alpha=0.5):
    # Fit
    ridge = Ridge(alpha)
    ridge.fit(X.iloc[train], y.iloc[train])

    # Predict
    prediction = ridge.predict(X.iloc[test])
    return prediction

def predict_LarsLasso(X, y, train, test, alpha=0.1):
    # Fit
    lars = LassoLars(alpha)
    lars.fit(X.iloc[train], y.iloc[train])

    # Predict
    prediction = lars.predict(X.iloc[test])
    return prediction
    
def predict_poly(X, y, train, test, degree=2):
    # Fit
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.iloc[train])
    X_poly_test = poly.fit_transform(X.iloc[test])
    linereg = LinearRegression()
    linereg.fit(X_poly, y.iloc[train])

    # Predict
    prediction = linereg.predict(X_poly_test)
    return prediction

predict_poly_2 = functools.partial(predict_poly, degree=2)
predict_poly_3 = functools.partial(predict_poly, degree=3)
predict_poly_4 = functools.partial(predict_poly, degree=4)


def chunkify(X, y, func, chunks):

    previous_rmse = 0
    previous_r2 = 0
    no_more_data = False
    for chunksize in chunks:
        if no_more_data:
            break

        print "chunksize: %s" % chunksize
        if len(X) < chunksize:
            print "Chunk size is too large!"
            print "Dataset X only has %s rows" % len(X)
            print "Using all %s rows for this chunk" % len(X)
            chunksize = len(X)
            no_more_data = True
        
        rmse, r2 = kfold(func, X, y, chunksize)
        delta_rmse = previous_rmse - rmse
        previous_rmse = rmse
        delta_r2 = previous_r2 - r2
        previous_r2 = r2
        print "rmse: %s   delta: %s" % (rmse, delta_rmse)
        print "r2: %s   delta: %s" % (r2, delta_r2)
        print ""


chunks = [100, 500, 1000, 5000, 10000, 50000, 100000]#, 500000]

Xlens, ylens = load_movielens()

datasets = [
    #("SUM without noise", Xquiet, yquiet),
    #("SUM with noise", Xnoisy, ynoisy),
    #("MovieLens", Xlens, ylens),
    ("Skin", Xlens, ylens)
]

for name, X, y in datasets:
    print "Using %s" % name
    print "LARS Lasso Regression"
    print "----------------"
    chunkify(X, y, functools.partial(predict_LarsLasso, alpha=0.5), chunks)   

    print "Linear Regression"
    print "-----------------"
    chunkify(X, y, predict_linear, chunks)



import pdb;pdb.set_trace()

