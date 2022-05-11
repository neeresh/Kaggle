import pandas as pd

def map_target(category):
    mappings = {'DOKOL': 0, 'SAFAVI': 1, 'ROTANA': 2, 'DEGLET': 3, 
                'SOGAY': 4, 'IRAQI': 5, 'BERHI': 6}
    
    return mappings[category]

dataset = pd.read_excel('Date_Fruit_Datasets.xlsx')

# Target Variable
print(dict(dataset['Class'].value_counts()))


# Missing Variables
print(dataset.isnull().sum())

# dataset.hist(bins = 30)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.drop(['Class'], axis=1),
                                                    dataset['Class'],
                                                    test_size = 0.3,
                                                    random_state = 24)

# Applying Standardisation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train, y_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print(y_train.value_counts())
# print(y_test.value_counts())

y_train = y_train.apply(map_target)
y_test = y_test.apply(map_target)

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state = 24, multi_class='ovr')
logit.fit(X_train, y_train)

pred_train = logit.predict(X_train)
pred_train_probs = logit.predict_proba(X_train)

pred_test = logit.predict(X_test)
pred_test_probs = logit.predict_proba(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score

print('train roc-auc: {}'.format(roc_auc_score(y_train, pred_train_probs, multi_class='ovr')))
print('train accuracy: {}'.format(accuracy_score(y_train, pred_train)))
print()
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred_test_probs, multi_class='ovr')))
print('test accuracy: {}'.format(accuracy_score(y_test, pred_test)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print('Confusion Matrix for train set: \n', confusion_matrix(y_train, pred_train))
print()
print('Confusion Matrix for test set: \n', confusion_matrix(y_test, pred_test))

# Plotting ROC-AUC
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(logit, classes=[0, 1, 2, 3, 4, 5, 6])
visualizer.fit(X_train, y_train)        
visualizer.score(X_test, y_test)        
visualizer.show()


