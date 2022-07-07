import pandas as pd
import numpy as np

# Importing dataset
dataset = pd.read_csv("train.csv")

# Feature Types
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == "object"]
numerical_features = [feature for feature in dataset.columns if feature not in categorical_features and feature != "Transported"]

# Numerical Features
skewed = ['Age']
extremely_skewed = [feature for feature in numerical_features if feature not in skewed]

# Splitting the Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['Transported'], axis=1),
    dataset['Transported'],
    test_size = 0.3,
    random_state = 12
    )

# Missing Values

    # Numerical Features
mean_value = X_train['Age'].mean() # Can try different metric other than mean

# X_train['Age'].replace(0, np.nan, inplace = True)
X_train['Age'] =  X_train['Age'].fillna(mean_value)

# X_test['Age'].replace(0, np.nan, inplace = True)
X_test['Age'] =  X_test['Age'].fillna(mean_value)

    # Handling Missing values in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in extremely_skewed:
    median_value = int(X_train[feature].median())
    
#    X_train[feature + '_na'] = np.where(X_train[feature].isnull(), 1, 0)
#    X_test[feature + '_na'] = np.where(X_test[feature].isnull(), 1, 0)
    
    X_train[feature] = X_train[feature].fillna(median_value)
    X_test[feature] = X_test[feature].fillna(median_value)

    # Categorical Features
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(random_state = 12)
random_imputer.fit(X_train[categorical_features])
    
X_train[categorical_features] = random_imputer.transform(X_train[categorical_features])
X_test[categorical_features] = random_imputer.transform(X_test[categorical_features])

print(X_train[categorical_features].isnull().sum())
print(X_test[categorical_features].isnull().sum())

"""
# Adding Additional Features
X_train.insert(1, 'GroupID', X_train['PassengerId'].apply(lambda x : int(x.split('_')[0])))
X_train.insert(2, 'PeopleInEachGroup', X_train['PassengerId'].apply(lambda x : int(x.split('_')[1])))

X_test.insert(1, 'GroupID', X_test['PassengerId'].apply(lambda x : int(x.split('_')[0])))
X_test.insert(2, 'PeopleInEachGroup', X_test['PassengerId'].apply(lambda x : int(x.split('_')[1])))

X_train.insert(6, 'Deck', X_train['Cabin'].apply(lambda x: str(x).split('/')[0]))
X_train.insert(7, 'DeckNum', X_train['Cabin'].apply(lambda x: int(x.split('/')[1])))
X_train.insert(8, 'DeckSide', X_train['Cabin'].apply(lambda x: str(x).split('/')[2]))

X_test.insert(6, 'Deck', X_test['Cabin'].apply(lambda x: str(x).split('/')[0]))
X_test.insert(7, 'DeckNum', X_test['Cabin'].apply(lambda x: int(x.split('/')[1])))
X_test.insert(8, 'DeckSide', X_test['Cabin'].apply(lambda x: str(x).split('/')[2]))
"""
"""
# Dropping Features
X_train.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
X_test.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
"""

# Change Additional Features datatypes

# Check Same Names: dataset['Name'].value_counts().head(21)
    # There are 20 names repeated.
        # 1. Drop Name columns
        # 2. Remove duplicate Names
X_train.drop(['Name', 'PassengerId'], axis=1, inplace=True)
X_test.drop(['Name', 'PassengerId'], axis=1, inplace=True)

# Encoding the Categorical Features

    # Monotonic Relationships Encoding

# Destination -> 3 unique categories -> One Hot Encoding of top categories (Top 2)
# Deck -> 8 unique categories -> OneHotEncoding of top categories (Top 5 or 6 or 7)

# HomePlanet -> 3 unique categories -> Weight Of Evidence (Test Set is not monotonic)
# CryoSleep -> 2 unique categories (Y/N) -> Weight Of Evidence (Monotonic)
# VIP -> 2 unique categories (Y/N) -> Weight Of Evidence (Monotonic)
# DeckSide -> 2 unique categories -> Weight Of Evidence (Monotonic)


"""
# Decision Tree Discretisation
from feature_engine.discretisation import DecisionTreeDiscretiser

to_decision_tree_disc = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

decision_tree_disc = DecisionTreeDiscretiser(cv = 5, scoring = 'roc_auc', 
                                               variables = to_decision_tree_disc,
                                              regression = False, 
                                               param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                             'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
decision_tree_disc.fit(X_train, y_train)

# print(decision_tree_disc.get_params())

X_train = decision_tree_disc.transform(X_train)
X_test = decision_tree_disc.transform(X_test)
"""

# Categorical Encoding
"""
    # Weight Of Evidence Encoding
to_woe = ['HomePlanet', 'CryoSleep', 'VIP']

from feature_engine.encoding import WoEEncoder

woe_encoder = WoEEncoder(variables = to_woe)
woe_encoder.fit(X_train, y_train)

X_train = woe_encoder.transform(X_train)
X_test = woe_encoder.transform(X_test)
"""

    # One Hot Encoder Top Categories
to_OHE_top_two = ['Destination']
# to_OHE_top_five = ['Deck']

from feature_engine.encoding import OneHotEncoder

destination_encoder = OneHotEncoder(top_categories=2, variables = to_OHE_top_two, 
                                    drop_last = True)
destination_encoder.fit(X_train)

X_train = destination_encoder.transform(X_train)
X_test = destination_encoder.transform(X_test)

remaining_categorical_features = ['HomePlanet', 'CryoSleep', 'VIP']
rcf = OneHotEncoder(variables = remaining_categorical_features, 
                                    drop_last = True)
rcf.fit(X_train)

X_train = rcf.transform(X_train)
X_test = rcf.transform(X_test)

X_train.drop(['Cabin'], axis = 1, inplace = True)
X_test.drop(['Cabin'], axis = 1, inplace = True)


print(X_train.isnull().sum())
print(X_test.isnull().sum())

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
# import joblib

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train.to_csv('xtrain.csv', index = False)
X_test.to_csv('xtest.csv', index = False)

y_train.to_csv('ytrain.csv', index = False)
y_test.to_csv('ytest.csv', index = False)

# joblib.dump(scaler, 'standard_scalar.joblib')
