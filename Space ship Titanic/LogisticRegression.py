# Lasso Selected Features
selected_features_lasso = pd.read_csv('selected_features_lasso.csv')
selected_features_lasso = selected_features_lasso['0'].to_list()

# X_train_lasso = X_train[selected_features_lasso]
# X_test_lasso = X_test[selected_features_lasso]

X_train_lasso = X_train
X_test_lasso = X_test

# 1.1 Logistic Regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=12, max_iter=500)
logit.fit(X_train_lasso, y_train.values.ravel())

pred_train_lasso = logit.predict_proba(X_train_lasso)[:, 1]
pred_test_lasso = logit.predict_proba(X_test_lasso)[:, 1]

from sklearn.metrics import roc_auc_score
print("ROC_AUC for Logistic Regression (TRAIN): ", roc_auc_score(y_train, pred_train_lasso))
print("ROC_AUC for Logistic Regression (TEST): ", roc_auc_score(y_test, pred_test_lasso))

    # Accuracy
from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_train, logit.predict(X_train_lasso)))
print("Test Accuracy: ", accuracy_score(y_test, logit.predict(X_test_lasso)))

    # Plotting ROC_AUC
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(logit, classes=['False', 'True'])
visualizer.fit(X_train_lasso, y_train)        
visualizer.score(X_test_lasso, y_test)        
visualizer.show()

# Hyperparameter Optimization

    # Grid Search

from sklearn.model_selection import GridSearchCV
logit_grid = LogisticRegression(random_state = 12, max_iter=500)

param_grid = dict(
    penalty = ['l1', 'l2'],
    C = [0.001, 0.01, 0.1, 1.0, 2.0],
    solver = ['liblinear']
    )

print('Number of Hyperparameter Combinations: ', len(param_grid['penalty'] * 
                                                     len(param_grid['C']) * len(param_grid['solver'])))

search = GridSearchCV(logit_grid, param_grid, scoring='accuracy', cv=10, refit=True)
search.fit(X_train_lasso, y_train.values.ravel())

print(search.best_params_)
results = pd.DataFrame(search.cv_results_)

pred_train_grid = search.predict_proba(X_train_lasso)[:, 1]
pred_test_grid = search.predict_proba(X_test_lasso)[:, 1]

from sklearn.metrics import roc_auc_score
print("ROC_AUC for Logistic Regression (TRAIN): ", roc_auc_score(y_train, pred_train_grid))
print("ROC_AUC for Logistic Regression (TEST): ", roc_auc_score(y_test, pred_test_grid))

from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_train, search.predict(X_train_lasso)))
print("Test Accuracy: ", accuracy_score(y_test, search.predict(X_test_lasso)))

from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(search, classes=['False', 'True'])
visualizer.fit(X_train_lasso, y_train)        
visualizer.score(X_test_lasso, y_test)        
visualizer.show()

    # Bayesian Optimisation
from skopt import BayesSearchCV

search = BayesSearchCV(estimator=logit_grid, search_spaces=param_grid, scoring='roc_auc', cv=10,
    n_iter=50, random_state=12, n_jobs=4, refit=True)

search.fit(X_train, y_train.values.ravel())

search.best_params_

search.best_score_

X_train_preds = search.predict_proba(X_train)[:, 1]
X_test_preds = search.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
print("ROC_AUC for Logistic Regression (TRAIN): ", roc_auc_score(y_train, X_train_preds))
print("ROC_AUC for Logistic Regression (TEST): ", roc_auc_score(y_test, X_test_preds))

from sklearn.metrics import accuracy_score
print("Train Accuracy: ", accuracy_score(y_train, search.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, search.predict(X_test)))

from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(search, classes=['False', 'True'])
visualizer.fit(X_train_lasso, y_train)        
visualizer.score(X_test_lasso, y_test)        
visualizer.show()
