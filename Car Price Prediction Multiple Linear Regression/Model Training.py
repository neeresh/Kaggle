import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')

y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

features = pd.read_csv('selected_features.csv')
features = features['0'].tolist()

X_train = X_train[features]
X_test = X_test[features]

lin_model = Lasso(alpha=0.000001, random_state=0)
lin_model.fit(X_train, y_train)

# Predictions
pred_train = lin_model.predict(X_train)

lmbda = -0.6629608715219251
y_train_original = np.exp(np.log(lmbda * y_train + 1) / lmbda)
pred_train = np.exp(np.log(lmbda * pred_train + 1) / lmbda)

# Training Set
print('train mse: {}'.format(int(mean_squared_error(y_train_original, pred_train))))
print('train rmse: {}'.format(int(mean_squared_error(y_train_original, pred_train, squared=False))))
print('train r2: {}'.format(r2_score(y_train_original, pred_train)))
print()

# Testing Set
pred_test = lin_model.predict(X_test)

y_test_original = np.exp(np.log(lmbda * y_test + 1) / lmbda)
pred_test = np.exp(np.log(lmbda * pred_test + 1) / lmbda)

# determine mse, rmse and r2
print('test mse: {}'.format(int(mean_squared_error(y_test_original, pred_test))))
print('test rmse: {}'.format(int(mean_squared_error(y_test_original, pred_test, squared=False))))
print('test r2: {}'.format(r2_score(y_test_original, pred_test)))
print()

print('Average car price: ', int(y_train_original.median()))


# Plotting
plt.scatter(y_test, lin_model.predict(X_test))
plt.xlabel('True Car Price')
plt.ylabel('Predicted Car Price')
plt.title('Evaluation of Lasso Predictions')

plt.show()

# Resetting index
y_test.reset_index(drop=True)
y_test.reset_index(drop=True, inplace=True)

preds = pd.Series(lin_model.predict(X_test))

# Plotting distribution of errors
errors = y_test['price'] - preds
errors.hist(bins=30)
plt.show()

# Important Features
importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = features
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance')

plt.show()



