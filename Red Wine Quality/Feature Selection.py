import pandas as pd

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

rf = RandomForestClassifier(
    n_estimators=10, random_state=1, n_jobs=4)

sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=rf,
    scoring="accuracy",
    cv=3)

sel.fit(X_train, y_train.values.ravel())

sel.feature_performance_

pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('accuracy')

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

# Saving X_train and X_test
X_train.to_csv("X_train_model_features.csv", index = False)
X_test.to_csv("X_test_model_features.csv", index = False)

