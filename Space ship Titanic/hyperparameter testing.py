# Feature Engineering
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
# Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from yellowbrick.classifier import ROCAUC


# Importing the train dataset
dataset = pd.read_csv('train.csv')

# Seperating the data into train and test
X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

y_train = y_train.astype(int).values.ravel()
y_test = y_test.astype(int).values.ravel()

# Imports
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

param_grid = [
    Integer(100, 3000, name = 'n_estimators'),
    Categorical(['gini', 'entropy'], name = 'criterion'),
    Integer(1, 30, name = 'max_depth'),
    Integer(2, 20, name = 'min_samples_split'),
    Integer(1, 20, name = 'min_samples_leaf'),
    Categorical(['auto', 'sqrt', 'log2'], name = 'max_features')
    ]

rf_classifier = RandomForestClassifier(random_state = 0)

import numpy as np

@use_named_args(param_grid)
def objective(**params):
    
    # model with new parameters
    rf_classifier.set_params(**params)

    # optimization function (hyperparam response function)
    value = np.mean(
        cross_val_score(
            rf_classifier, 
            X_train,
            y_train,
            cv=3,
            n_jobs=-1,
            scoring='accuracy')
    )

    # negate because we need to minimize
    return -value

gp_ = gp_minimize(
    objective, # the objective function to minimize
    param_grid, # the hyperparameter space
    n_initial_points=10, # the number of points to evaluate f(x) to start of
    acq_func='EI', # the acquisition function
    n_calls=50, # the number of subsequent evaluations of f(x)
    random_state=0, 
)

"Best score=%.4f" % gp_.fun # 'Best score=-0.7977' -> [322, 'entropy', 10, 4, 1, 'sqrt']

gp_.x # list of all best parameters

plot_convergence(gp_)


"""
import xgboost as xgb


param_grid = [
    Integer(200, 2500, name='n_estimators'),
    Integer(1, 10, name='max_depth'),
    Real(0.01, 0.99, name='learning_rate'),
    Categorical(['gbtree', 'dart'], name='booster'),
    Real(0.01, 10, name='gamma'),
    Real(0.50, 0.90, name='subsample'),
    Real(0.50, 0.90, name='colsample_bytree'),
    Real(0.50, 0.90, name='colsample_bylevel'),
    Real(0.50, 0.90, name='colsample_bynode'),
    Integer(1, 50, name='reg_lambda'),
]

gbm = xgb.XGBClassifier(random_state=1000)

import numpy as np

@use_named_args(param_grid)
def objective(**params):
    
    # model with new parameters
    gbm.set_params(**params)

    # optimization function (hyperparam response function)
    value = np.mean(
        cross_val_score(
            gbm, 
            X_train,
            y_train,
            cv=3,
            n_jobs=-4,
            scoring='accuracy')
    )

    # negate because we need to minimize
    return -value


gp_ = gp_minimize(
    objective, # the objective function to minimize
    param_grid, # the hyperparameter space
    n_initial_points=10, # the number of points to evaluate f(x) to start of
    acq_func='EI', # the acquisition function
    n_calls=50, # the number of subsequent evaluations of f(x)
    random_state=0, 
)

"Best score=%.4f" % gp_.fun # 'Best score=-0.8061'

gp_.x # list of all best parameters

print(""Best parameters:
=========================
- n_estimators = %d
- max_depth = %d
- learning_rate = %.6f
- booster = %s
- gamma = %.6f
= subsample = %.6f
- colsample_bytree = %.6f
- colsample_bylevel = %.6f
- colsample_bynode' = %.6f
"" % (gp_.x[0],
       gp_.x[1],
       gp_.x[2],
       gp_.x[3],
       gp_.x[4],
       gp_.x[5],
       gp_.x[6],
       gp_.x[7],
       gp_.x[8],
      ))
    
"""
"""
Best parameters:
=========================
- n_estimators = 1049
- max_depth = 10
- learning_rate = 0.226376
- booster = dart
- gamma = 9.688593
= subsample = 0.875223
- colsample_bytree = 0.583697
- colsample_bylevel = 0.500000
- colsample_bynode' = 0.900000
"""
"""
plot_convergence(gp_)

"""




