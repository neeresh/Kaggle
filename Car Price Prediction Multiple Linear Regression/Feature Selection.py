import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

dataset = pd.read_csv('CarPrice_Assignment.csv')

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

# Feature Selection
sel_ = SelectFromModel(Lasso(alpha=[0.000001], random_state=0))
sel_.fit(X_train, y_train)

# To see selected features
selected_features = X_train.columns[(sel_.get_support())]
print(sel_.get_support().sum())

# Saving features in the csv file
pd.Series(selected_features).to_csv('selected_features.csv', index=False)

# Heat map
sns.heatmap(dataset.corr(), annot=True, linewidths=.5)
sns.set(rc = {'figure.figsize':(15, 8)})

# ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 
#  'horsepower', 'citympg', 'highwaympg']




