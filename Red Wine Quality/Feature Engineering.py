import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import scipy.stats as stats

dataset = pd.read_csv('winequality-red.csv')

to_log_tranformation = ['chlorides', 'total sulfur dioxide', 'pH', 'sulphates']
to_boxcox = ['fixed acidity', 'volatile acidity']
to_exp = ['density']

dataset = dataset.astype({"quality": "O"})

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O' and feature != 'quality']

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['quality'], axis=1),
    dataset['quality'],
    test_size = 0.4,
    random_state = 0
    )

# Log Transformation
for feature in to_log_tranformation:
    X_train[feature] = np.log(X_train[feature])
    X_test[feature] = np.log(X_test[feature])

"""
# fold = 3
to_gaussian_capping = ['pH', 'sulphates']

# fold = 1.5
to_iqr = ['chlorides']


# Treating Outliers - 1
from feature_engine.outliers import Winsorizer

windsoriser_gaussian = Winsorizer(capping_method='gaussian', 
                          tail='both',  
                          fold = 3,
                          variables=to_gaussian_capping)

windsoriser_iqr = Winsorizer(capping_method='iqr', 
                          tail='both',  
                          fold = 1.5,
                          variables=to_iqr)

windsoriser_gaussian.fit(X_train)
X_train = windsoriser_gaussian.transform(X_train)
X_test = windsoriser_gaussian.transform(X_test)

windsoriser_iqr.fit(X_train)
X_train = windsoriser_iqr.transform(X_train)
X_test = windsoriser_iqr.transform(X_test)
"""

# Box Cox Transformation
lmbdas = {}

for feature in to_boxcox:
    X_train[feature], lmbda = stats.boxcox(X_train[feature])
    lmbdas[feature] = lmbda

for feature in to_boxcox:
    X_test[feature] = stats.boxcox(X_test[feature], lmbda=lmbdas[feature])

# Exponential Transformation
X_train[to_exp] = np.exp(X_train[to_exp])
X_test[to_exp] = np.exp(X_test[to_exp])

# Equal Frequency Discretisation
from feature_engine.discretisation import EqualFrequencyDiscretiser

to_efd_q5 = ['citric acid']
to_efd_q6 = ['alcohol']

disc_citric_acid = EqualFrequencyDiscretiser(q = 5, variables = to_efd_q5)
disc_citric_acid.fit(X_train)

X_train = disc_citric_acid.transform(X_train)
X_test = disc_citric_acid.transform(X_test)

# print(disc_critic_acid.binner_dict_)

disc_alcohol = EqualFrequencyDiscretiser(q = 6, variables = to_efd_q6)
disc_alcohol.fit(X_train)

# print(disc_alcohol.binner_dict_)

X_train = disc_alcohol.transform(X_train)
X_test = disc_alcohol.transform(X_test)

# Residual Sugar: Treat Outliers as missing values then perform missing imputation
X_train['residual sugar'] = np.where(X_train['residual sugar'] < 5.5, X_train['residual sugar'], np.nan)
X_test['residual sugar'] = np.where(X_test['residual sugar'] < 5.5, X_test['residual sugar'], np.nan)

X_train['residual sugar'] = X_train['residual sugar'].fillna(round(X_train['residual sugar'].mean(), 1))
X_test['residual sugar'] = X_test['residual sugar'].fillna(round(X_test['residual sugar'].mean(), 1))

# Log Transformation
X_train['residual sugar'] = np.log(X_train['residual sugar'])
X_test['residual sugar'] = np.log(X_test['residual sugar'])

# Free Sulfur Dioxide: Treat Outliers as missing values then perform missing imputation
X_train['free sulfur dioxide'] = np.where(X_train['free sulfur dioxide'] < 45, X_train['free sulfur dioxide'], np.nan)
X_test['free sulfur dioxide'] = np.where(X_test['free sulfur dioxide'] < 45, X_test['free sulfur dioxide'], np.nan)

X_train['free sulfur dioxide'] = X_train['free sulfur dioxide'].fillna(round(X_train['free sulfur dioxide'].mean()))
X_test['free sulfur dioxide'] = X_test['free sulfur dioxide'].fillna(round(X_test['free sulfur dioxide'].mean()))

# Log Transformation
X_train['free sulfur dioxide'] = np.log(X_train['free sulfur dioxide'])
X_test['free sulfur dioxide'] = np.log(X_test['free sulfur dioxide'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(X_train)

X_train = pd.DataFrame(standard_scaler.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(standard_scaler.transform(X_test), columns = X_test.columns)

# Saving the train and test data for next steps
X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)

y_train.to_csv('y_train.csv', index = False)
y_test.to_csv('y_test.csv', index = False)

# Saving the joblib
import joblib

joblib.dump(standard_scaler, 'standard_scaler.joblib')
