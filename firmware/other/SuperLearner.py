from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np

# example of a super learner model for binary classification


# create a list of base-models
def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    models.append(BaggingRegressor(n_estimators=10))
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    return models


# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=10, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
    # get data
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    meta_y.extend(test_y)
    # fit and make predictions with each sub-model
    for model in models:
        model.fit(train_X, train_y)
    yhat = model.predict(test_X)
    # store columns
    fold_yhats.append(yhat.reshape(len(yhat), 1))
    # store fold yhats as columns
    meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)


# fit all base models on the training dataset
def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)


# fit a meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
    mse = mean_squared_error(y, yhat)
    print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))


# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
    meta_X.append(yhat.reshape(len(yhat), 1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)


# create the inputs and outputs
#X, y = make_regression(n_samples=1000, n_features=100, noise=0.5)

X_read = pd.read_csv('X.csv')
y_read = pd.read_csv('y.csv')

X_cols = X_read.columns.tolist()
X_dict = {}
y = y_read['pm2_5Palas'].tolist()
y_arr_list = []

for element in X_cols:
    X_dict[element] = X_read[element].tolist()

for k in X_dict:
    print(k)

X_arr = np.array([X_dict['NH3_loRa'], X_dict['CO_loRa'], X_dict['NO2_loRa'], X_dict['C3H8_loRa'], X_dict['C4H10_loRa'], X_dict['CH4_loRa'], X_dict['H2_loRa'],
                 X_dict['C2H5OH_loRa'], X_dict['P1_lpo_loRa'], X_dict['P1_ratio_loRa'], X_dict['P1_conc_loRa'], X_dict['P2_lpo_loRa'], X_dict['P2_ratio_loRa'],
                  X_dict['P2_conc_loRa'], X_dict['Temperature_loRa'], X_dict['Pressure_loRa'], X_dict['Humidity_loRa']])

y_arr = np.array([y,])
X_arr = X_arr.T
y_arr = y_arr.T

X, X_val, y, y_val = train_test_split(X_arr, y_arr, test_size=0.20)

# X, y = make_regression(n_samples=1000, n_features=100, noise=0.5)
# print(y.shape)
# X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)

print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)

models = get_models()
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
print('Meta ', meta_X.shape, meta_y.shape)

fit_base_models(X, y, models)
meta_model = fit_meta_model(meta_X, meta_y)
evaluate_models(X_val, y_val, models)
yhat = super_learner_predictions(X_val, models, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))
