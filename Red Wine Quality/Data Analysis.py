import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

dataset = pd.read_csv('winequality-red.csv')

# Missing Values
print(dataset.isnull().sum())

# Variable Types
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O' and feature != 'quality']

for feature in numerical_features:
    print(feature, '-->' , dataset[feature].nunique())

# Treat Outliers
def plot_outliers(dataset, feature):
    
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(dataset[feature], bins=30).set(title = "Histogram")
    
    plt.subplot(1, 3, 2)
    stats.probplot(dataset[feature], dist = 'norm', plot = plt)
    
    plt.subplot(1, 3, 3)
    sns.boxplot(y = dataset[feature]).set(title = "Box plot")
    
    plt.show()

for feature in numerical_features:
    plot_outliers(dataset, feature)

# Plot kde
def plot_kde(dataset, feature=None, x_label=None):
    
    if feature != None:
        sns.displot(dataset[feature], kind='kde')
    
    else:
        sns.displot(dataset, kind='kde')
        plt.xlabel(x_label)

# Plot Before and After transformation
def plot_before_after(dataset, feature):
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data = dataset, x=feature, y='quality')
    
    plt.subplot(1, 2, 2)
    stats.probplot(dataset[feature], dist='norm', plot=plt)
    
    plt.show()
        
# for feature in numerical_features:
#     plot_before_after(dataset, feature)

for feature in numerical_features:
    plot_kde(dataset, feature)


# Log transformation
for feature in numerical_features:
    print("Log Transform: ", feature)
    plot_kde(np.log(dataset[numerical_features]), feature)

to_log_tranformation = ['chlorides', 'total sulfur dioxide', 'pH', 'sulphates']

# Transforming to Log - 1
for feature in to_log_tranformation:
    dataset[feature] = np.log(dataset[feature])

# Checking Outliers in to_log_transformation features - 1
for feature in to_log_tranformation:
    plot_outliers(dataset, feature)

# Treating Outliers - 1
from feature_engine.outliers import Winsorizer

outliers_log_transformation = ['chlorides', 'sulphates']
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both',  
                          fold = 3,
                          variables=outliers_log_transformation)

windsoriser.fit(dataset)
dataset = windsoriser.transform(dataset)

# Checking Outliers in to_log_transformation features - 1
for feature in outliers_log_transformation:
    plot_outliers(dataset, feature)

# fold = 3
to_gaussian_capping = ['pH', 'sulphates']

# fold = 1.5
to_iqr = ['chlorides']

# Squareroot transformation
for feature in numerical_features:
    plot_kde(np.sqrt(dataset[numerical_features]), feature)

# Yeo-johnson transformation
for feature in numerical_features:
    fitted_data, _ = stats.yeojohnson(dataset[feature])
    plot_kde(fitted_data, x_label=feature)

# Boxcox transformation
for feature in numerical_features:
    fitted_data, _ = stats.boxcox(dataset[feature])
    plot_kde(fitted_data, x_label=feature)

to_boxcox = ['fixed acidity', 'volatile acidity']

# Exponential transformation
for feature in numerical_features:
    plot_kde(np.exp(dataset[numerical_features]), feature)

to_exp = ['density']

# Transforming to exp - 1
dataset[to_exp] = np.exp(dataset[to_exp])

# Checking Outliers - 1
plot_outliers(dataset, 'density')

# Treating Outliers - 1
from feature_engine.outliers import Winsorizer

windsoriser = Winsorizer(capping_method='iqr', 
                          tail='both',  
                          fold = 1.5,
                          variables='density')

windsoriser.fit(dataset)
dataset = windsoriser.transform(dataset)

# Checking Outliers - 1
plot_outliers(dataset, 'density')

# Inverse Transformation
for feature in numerical_features:
    plot_kde(1/dataset, feature)


remaining_features = [feature for feature in numerical_features if feature not in 
                      to_log_tranformation and feature not in to_boxcox and 
                      feature not in to_exp]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset[remaining_features],
    dataset['quality'],
    test_size = 0.3,
    random_state = 0
    )

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

# Equal Frequency discretisation
# q = 6 ['alcohol]
# q = 5 ['citric acid']
for feature in remaining_features:
    plot_discretisation(X_train, X_test, 'efd', feature, q=5)

to_efd_q5 = ['citric acid']
to_efd_q6 = ['alcohol']

remaining_features = [feature for feature in remaining_features 
                      if feature not in to_efd_q5 and feature not in to_efd_q6]

# Equal width discretisation
for feature in remaining_features:
    plot_discretisation(X_train, X_test, 'ewd', feature, bins=5)

# K-means discretisation
for feature in remaining_features:
    plot_discretisation(X_train, X_test, 'kmeans', feature, bins=50)

sns.heatmap(dataset.corr(), annot=True, linewidths=1.5)
sns.set(rc={'figure.figsize': (8, 7)})

# Drop : ['residual sugar,' 'free sulfur dioxide']

# OR

# Treating outliers
from feature_engine.outliers import Winsorizer

def plot_outliers(dataset, method, features, fold = 0.05, cap_to = 'both'):
    
    windsoriser = Winsorizer(capping_method=method, 
                          tail=cap_to, 
                          fold=fold,
                          variables=features)

    windsoriser.fit(dataset)
    windsoriser.transform(dataset)
    
    plot_kde(dataset[features])

# 1. Residual Sugar: Treat outliers as missing values and then replace by mean
sns.displot(dataset['residual sugar'], kind='kde')

plot_kde(np.log(dataset[dataset['residual sugar'] < 5.5]['residual sugar']))

a = pd.DataFrame(np.where(dataset['residual sugar'] < 5.5, dataset['residual sugar'], np.nan))
a.columns = ['residual sugar']

plot_kde(a)

plot_kde(np.log(a)) # with nan

a.fillna(round(a['residual sugar'].mean(), 1), inplace=True)

plot_kde(np.log(a))

# 2. free sulfur dioxide Treat outliers as missing values and then replace by mean
plot_kde(dataset['free sulfur dioxide'])

plot_kde(np.sqrt(dataset[dataset['free sulfur dioxide'] < 45]['free sulfur dioxide']))
stats.probplot(np.sqrt(dataset[dataset['free sulfur dioxide'] < 45]['free sulfur dioxide']), dist='norm', plot=plt)

b = pd.DataFrame(np.where(dataset['free sulfur dioxide'] < 45, dataset['free sulfur dioxide'], np.nan))
b.columns = ['free sulfur diocide']

plot_kde(b)
plot_kde(np.sqrt(b))

b.fillna(round(b['free sulfur diocide'].mean()), inplace=True)

plot_kde(np.sqrt(b))





