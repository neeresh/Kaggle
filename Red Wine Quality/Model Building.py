import pandas as pd

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Over-sampling dataset
from imblearn.over_sampling import SMOTE

sm_rf = SMOTE(sampling_strategy='auto',
              random_state=1, k_neighbors=5)
X_train, y_train = sm_rf.fit_resample(X_train, y_train)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty = 'l2', random_state=1, multi_class="multinomial", max_iter=300, class_weight='balanced')

classifier.fit(X_train, y_train.values.ravel())

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

y_pred_train_probs = classifier.predict_proba(X_train)
y_pred_test_probs = classifier.predict_proba(X_test)

from sklearn.metrics import roc_auc_score
print('ROC-AUC Random Forest train:', roc_auc_score(y_train, y_pred_train_probs, multi_class='ovr'))
print('ROC-AUC Random Forest test:', roc_auc_score(y_test, y_pred_test_probs, multi_class='ovr'))

# Plotting ROC-AUC
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(
    classifier, per_class=True, cmap="cool", micro=False,
)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

