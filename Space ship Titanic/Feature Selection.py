import pandas as pd
import numpy as np

from mlxtend.feature_selection import ExhaustiveFeatureSelector

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

y_train = y_train.astype(int)
y_test = y_test.astype(int)


all_features = pd.DataFrame(X_train.columns.to_list())
all_features.to_csv('all_features.csv', index=False)

# Exhaustive Search
"""
# Exhaustive Feature Selection
from sklearn.ensemble import RandomForestClassifier
efs = ExhaustiveFeatureSelector(
    RandomForestClassifier(n_estimators = 100, n_jobs = 4, random_state = 12, max_depth = None),
    min_features = 1,
    max_features = 10,
    scoring = 'roc_auc',
    print_progress  = True,
    cv = 5
    )

efs = efs.fit(np.array(X_train), y_train.values.ravel())

print(efs.best_idx_)
print(X_train[efs.best_idx_])
"""

# Step-backward Feature Selection
"""
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

sfs = SequentialFeatureSelector(
    RandomForestClassifier(n_estimators=100, random_state=12),
    k_features=20,
    forward=False,
    floating=False,
    scoring='roc_auc',
    cv = 5
    )

sfs.fit(np.array(X_train), y_train.values.ravel())

print(list(sfs.k_feature_idx_))

X_train_sbs = X_train.columns[list(sfs.k_feature_idx_)]
X_test_sbs = X_test.columns[list(sfs.k_feature_idx_)]

selected_features_sbs_index = pd.DataFrame(list(sfs.k_feature_idx_))
selected_features_sbs_index.to_csv('selected_features_sbs_index.csv', index=False)

select_features_sbs_names = pd.DataFrame(X_train.columns[list(sfs.k_feature_idx_)])
select_features_sbs_names.to_csv('select_features_sbs_names.csv', index=False)
"""

# Lasso
"""
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(
    LogisticRegression(C = 0.5, penalty = 'l1', solver = 'liblinear', random_state = 12)
    )
sel_.fit(X_train, y_train)

selected_features_lasso = pd.DataFrame(X_train.columns[(sel_.get_support())])
selected_features_lasso.to_csv('selected_features_lasso.csv', index=False)
"""

