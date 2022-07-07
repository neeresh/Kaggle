import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

y_train = y_train.astype(int).values.ravel()
y_test = y_test.astype(int).values.ravel()

# 1.1 Logistic Regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=12, max_iter=500)
logit.fit(X_train, y_train.values.ravel())

pred_train = logit.predict_proba(X_train)[:, 1]
pred_test = logit.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, accuracy_score
print("ROC_AUC for Logistic Regression (TRAIN): ", roc_auc_score(y_train, pred_train))
print("Train Accuracy: ", accuracy_score(y_train, logit.predict(X_train)))
print()
print("ROC_AUC for Logistic Regression (TEST): ", roc_auc_score(y_test, pred_test))
print("Test Accuracy: ", accuracy_score(y_test, logit.predict(X_test)))


# Confusion Matrix
y_pred = logit.predict(X_test)
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, logit.predict(X_train)))

# Applying XGBoost
# from xgboost import XGBClassifier
# classifier = XGBClassifier()

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, classifier.predict(X_train)))

print("ROC_AUC for XGBoost (Train): ", roc_auc_score(y_train, classifier.predict(X_train)))
print("ROC_AUC for XGBoost (TEST): ", roc_auc_score(y_test, classifier.predict(X_test)))
print()
print("Train Accuracy: ", accuracy_score(y_train, classifier.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, classifier.predict(X_test)))
