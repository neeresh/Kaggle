import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import joblib

dataset = pd.read_csv('CarPrice_Assignment.csv')

dataset['symboling'] = dataset['symboling'].astype('object')

categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'object']
numerical_features = [feature for feature in dataset.columns if feature not in categorical_features and feature != 'price']

# When we engineer features, some techniques learn parameters from data. 
# It is important to learn those parameters only from the train set. This is to avoid overfit.

X_train, X_test, y_train, y_test = train_test_split(dataset.drop(['car_ID', 'price'], axis=1),
                                                    dataset['price'],
                                                    test_size = 0.2,
                                                    random_state=0
                                                    )

# Target Variable
fitted_data, lmbda = stats.boxcox(y_train)
print(lmbda)

index = y_train.index.values
y_train = pd.Series(fitted_data, name='price', index=index)

index = y_test.index.values
y_test = pd.Series(stats.boxcox(y_test, lmbda=lmbda), name='price', index=index)

# Numerical Variable Transformation
to_yeojohnson_transformation = ['carlength', 'carheight', 'curbweight', 'stroke', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

feature_lmbda = {}

for feature in to_yeojohnson_transformation:
    X_train[feature], lmbda  = stats.yeojohnson(X_train[feature])
    feature_lmbda[feature] = lmbda
    X_test[feature] = stats.yeojohnson(X_test[feature], lmbda=lmbda)

to_sqrt_transformation = ['boreratio']
X_train[to_sqrt_transformation] = np.sqrt(X_train[to_sqrt_transformation])
X_test[to_sqrt_transformation] = np.sqrt(X_test[to_sqrt_transformation])

to_inverse_transformation = ['enginesize']
X_train[to_inverse_transformation] = 1 / X_train[to_inverse_transformation]
X_test[to_inverse_transformation] = 1 / X_test[to_inverse_transformation]

to_log_transformation = ['wheelbase', 'carwidth']
for feature in to_log_transformation:
    X_train[feature] = np.log(X_train[feature])
    X_test[feature] = np.log(X_test[feature])

# Binarize extremely skewed features
to_binary_transformation = ['compressionratio']
X_train[to_binary_transformation] = np.where(X_train[to_binary_transformation] > 9.0, 0, 1)
X_test[to_binary_transformation] = np.where(X_test[to_binary_transformation] > 9.0, 0, 1)

# Categorical Features

# Categorical Mapping of CarName
def carname_to_company_name(dataset):
    
    car_company_names = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 
                         'isuzu', 'jaguar', 'mazda', 'buick', 'mercury cougar', 
                         'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 
                         'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'vw', 
                         'volvo']
        
    for i in dataset.index.values:
        for car_company in car_company_names:
            if car_company in dataset['CarName'][i].lower():
                dataset['CarName'] = dataset['CarName'].replace([dataset['CarName'][i]], car_company)
    
    return dataset


X_train = carname_to_company_name(X_train)
X_test = carname_to_company_name(X_test)

# Removing Rare Labels
def rare_lables(dataset, feature, rare_percentage):
    data = dataset.copy(deep = True)
    
    temp_data = data.groupby(feature)[feature].count() / len(data)
    
    return temp_data[temp_data > rare_percentage].index 

for feature in categorical_features:
    frequent_list = rare_lables(X_train, feature, 0.01)
    
    X_train[feature] = np.where(X_train[feature].isin(frequent_list), X_train[feature], 'Rare')
    X_test[feature] = np.where(X_test[feature].isin(frequent_list), X_test[feature], 'Rare')


# Encoding categorical variables (Before encoding achieveing monotonic relationship)
def replace_categories(train, test, y_train, categorical_feature, target_column_name):
    data = pd.concat([X_train, y_train], axis=1)
    
    ordered_labels = data.groupby([categorical_feature])[target_column_name].mean().sort_values().index    
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    
    print(feature, "--->", ordinal_label)
    print()
    
    train[feature] = train[feature].map(ordinal_label)
    test[feature] = test[feature].map(ordinal_label)

for feature in categorical_features:
    replace_categories(X_train, X_test, y_train, feature, 'price')

X_test['enginetype'].fillna(0, inplace=True)

# Checking for monotomic relationship
def check_monotonic_relationship(train, y_train, feature):
    temp_data = pd.concat([train, y_train], axis=1)
    
    temp_data.groupby(feature)['price'].median().plot.bar()
    plt.title(feature)
    
    plt.show()

for feature in categorical_features:
    check_monotonic_relationship(X_train, y_train, feature)


# Feature Scaling
scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

# Saving X_train, X_test, y_train, y_test
X_train.to_csv('xtrain.csv', index = False)
X_test.to_csv('xtest.csv', index = False)

y_train.to_csv('ytrain.csv', index = False)
y_test.to_csv('ytest.csv', index = False)


# Saving the Scaler
joblib.dump(scaler, 'minmax_scalar.joblib')




