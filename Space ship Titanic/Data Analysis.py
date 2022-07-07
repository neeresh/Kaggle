import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("train.csv")

# Percentage of Missing Values
ax = dataset.isnull().mean().plot.bar(figsize=(10, 5))
ax.set_yticklabels(['{:,.2%}'.format(x) for x in list(ax.get_yticks())])
plt.axhline(y=0.02, color = 'red')

plt.show()

# Target Variable
dataset['Transported'].value_counts()

# Ratio of binary classes
dataset['Transported'].value_counts() / len(dataset)

# Variable Types
dataset.info()

categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == "object"]
numerical_features = [feature for feature in dataset.columns if feature not in categorical_features and feature != "Transported"]

# Counting Unique values
def count_unique(df, feature, ascending = False):
    
    result_df = pd.DataFrame.from_dict(
        df[feature].value_counts().sort_values(ascending=False).to_dict(), orient = 'index')
    
    result_df = result_df.rename_axis(feature).reset_index()
    result_df.rename(columns = {0: 'value_counts'}, inplace = True)
    
    return result_df

Age_df = count_unique(dataset, 'Age')
RoomService_df = count_unique(dataset, 'RoomService') 
FoodCourt_df = count_unique(dataset, 'FoodCourt')

ShoppingMall_df = count_unique(dataset, 'ShoppingMall')
Spa_df = count_unique(dataset, 'Spa')
VRDeck_df = count_unique(dataset, 'VRDeck')

extremely_skewed = [feature for feature in numerical_features if feature != 'Age']

# Numerical Features
dataset[numerical_features].hist(figsize=(8, 8), bins=30)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['Transported'], axis=1),
    dataset['Transported'],
    test_size = 0.3,
    random_state = 12
    )

# dataset[numerical_features].isnull().sum()

mean_value = int(X_train['Age'].mean())

X_train['Age'].replace(0, np.nan, inplace = True)
X_train['Age'] =  X_train['Age'].fillna(mean_value)

X_test['Age'].replace(0, np.nan, inplace = True)
X_test['Age'] =  X_test['Age'].fillna(mean_value)

for feature in extremely_skewed:
    median_value = int(X_train[feature].median())
    
    X_train[feature] = X_train[feature].fillna(median_value)
    X_test[feature] = X_test[feature].fillna(median_value)

# Discretisation
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.discretisation import EqualWidthDiscretiser
from sklearn.preprocessing import KBinsDiscretizer

def plot_discretisation(X_train, X_test, method, feature, q=10, bins=10):
    
    if method == 'efd':  
        disc = EqualFrequencyDiscretiser(q = q, variables = feature)
    elif method == 'ewd':
        disc = EqualWidthDiscretiser(bins = bins, variables = feature)
    elif method == 'kmeans':
        disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    
    if method == 'kmeans':
        train_columns = X_train.columns
        test_columns = X_test.columns
    
    disc.fit(X_train)

    X_train = disc.transform(X_train)
    X_test = disc.transform(X_test)
    
    if method == 'kmeans':
        X_train = pd.DataFrame(X_train, columns=train_columns)
        X_test = pd.DataFrame(X_test, columns=test_columns)

    temp_train = X_train.groupby(feature)[feature].count() / len(X_train)
    temp_test = X_test.groupby(feature)[feature].count() / len(X_test)

    temp = pd.concat([temp_train, temp_test], axis=1)
    temp.columns = ['train', 'test']

    temp.plot.bar()
    plt.ylabel('Number of observations per bin')

plot_discretisation(X_train, X_test, method='efd', feature = 'Age', q = 7)

to_efd_q7 = ['Age']

from feature_engine.discretisation import DecisionTreeDiscretiser

def check_decision_tree_disc(X_train, X_test, feature, show_bins = False, inplace=False, plot_train = False, plot_test = False):
    
    treeDisc = DecisionTreeDiscretiser(cv=10, scoring='accuracy',
                                   variables=feature,
                                   regression=False,
                                   param_grid={'max_depth': [1, 2, 3],
                                              'min_samples_leaf':[10,4]})

    treeDisc.fit(X_train, y_train)

    print("Feature Scores: ", treeDisc.scores_dict_)
    
    if not isinstance(feature, list) and len(feature.split()) == 1 :
        feature = feature.split()
            
    X_train = treeDisc.transform(X_train)
    X_test = treeDisc.transform(X_test)
    
    # Checking How many bins
    if show_bins:
        for single_feature in feature:
            print("Total no.of bins in {}: {}".format(feature, X_train[single_feature].nunique()))
            print("{} Bins: ".format(single_feature), X_train[single_feature].unique())

    # Checking monotonic relationship
    if plot_train:
        for single_feature in feature:
            pd.concat([X_train, y_train], axis=1).groupby([single_feature])['Transported'].mean().plot()
            plt.title('Train Set Monotonic relationship between discretised {} and target'.format(single_feature))
            plt.ylabel('Transported')
            
            plt.show()
            
    if plot_test:
        for single_feature in feature:
            pd.concat([X_test, y_test], axis=1).groupby([single_feature])['Transported'].mean().plot()
            plt.title('Test Set Monotonic relationship between discretised {} and target'.format(single_feature))
            plt.ylabel('Transported')
            
            plt.show()

check_decision_tree_disc(X_train, X_test, feature = extremely_skewed, show_bins=True, inplace=True,
                         plot_train = True, plot_test = True)

# Categorical Features
print(categorical_features)

# Splitting PassengerId
X_train.insert(1, 'GroupID', X_train['PassengerId'].apply(lambda x : int(x.split('_')[0])))
X_train.insert(2, 'PeopleInEachGroup', X_train['PassengerId'].apply(lambda x : int(x.split('_')[1])))

X_test.insert(1, 'GroupID', X_test['PassengerId'].apply(lambda x : int(x.split('_')[0])))
X_test.insert(2, 'PeopleInEachGroup', X_test['PassengerId'].apply(lambda x : int(x.split('_')[1])))

# Cabin 
print(dataset['Cabin'].apply(lambda x: str(x).split('/')))

from feature_engine.imputation import RandomSampleImputer
cabin_imputer = RandomSampleImputer(random_state = 12)

cabin_imputer.fit(X_train[categorical_features]) 
print(cabin_imputer.variables_)

X_train[categorical_features] = cabin_imputer.transform(X_train[categorical_features])
X_test[categorical_features] = cabin_imputer.transform(X_test[categorical_features])

X_train.insert(6, 'Deck', X_train['Cabin'].apply(lambda x: str(x).split('/')[0]))
X_train.insert(7, 'DeckNum', X_train['Cabin'].apply(lambda x: int(str(x).split('/')[1])))
X_train.insert(8, 'DeckSide', X_train['Cabin'].apply(lambda x: str(x).split('/')[2]))

X_test.insert(6, 'Deck', X_test['Cabin'].apply(lambda x: str(x).split('/')[0]))
X_test.insert(7, 'DeckNum', X_test['Cabin'].apply(lambda x: int(x.split('/')[1])))
X_test.insert(8, 'DeckSide', X_test['Cabin'].apply(lambda x: str(x).split('/')[2]))


# Decretisation
to_efd_07 = ['DeckNum']
to_efd_09 = ['GroupID']

plot_discretisation(X_train, X_test, method = 'efd', feature = 'DeckNum', q = 7)
plot_discretisation(X_train, X_test, method = 'efd', feature = 'GroupID', q = 9)

X_train.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)
X_test.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)

all_categorical_features = [feature for feature in X_train.columns if feature not in numerical_features]

print(X_train[all_categorical_features].isnull().any())
print(X_test[all_categorical_features].isnull().any())

group_id_series = X_train['GroupID'].value_counts()
people_in_each_group_series = X_train['PeopleInEachGroup'].value_counts()

import seaborn as sns

sns.countplot(X_train['PeopleInEachGroup'], hue = y_train)
sns.countplot(X_train['DeckSide'], hue = y_train)
sns.countplot(X_train['Deck'], hue = y_train)

# Destination
sns.countplot(dataset['Destination'], hue=dataset['Transported'])

sns.boxplot(y = dataset['RoomService'])





