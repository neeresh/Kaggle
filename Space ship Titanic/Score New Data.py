# Feature Engineering
import pandas as pd

from sklearn.model_selection import train_test_split
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, RandomSampleImputer
from feature_engine.selection import DropFeatures
from feature_engine.discretisation import EqualFrequencyDiscretiser, DecisionTreeDiscretiser
from feature_engine.encoding import WoEEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from HelperMethods import ReplaceZerosWithNaN, MeanImputer, AdditionalFeatures

# Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ROCAUC

from sklearn.pipeline import Pipeline

import xgboost as xgb

# Importing the train dataset
dataset = pd.read_csv('train.csv')

# Seperating the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['Transported'], axis=1),
    dataset['Transported'],
    test_size = 0.3,
    random_state = 9999
    )

# Target - Converting 0 - False and 1 - True
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Configuration

    # Numerical Imputation
MEAN_IMPUTER = ['Age']
BINARY_AND_MEDIAN_IMPUTER = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Categorical Imputation
RANDOM_IMPUTER = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']

    # Additional Features and are ordered as in trianing set
ALL_FEATURES_ORDERED = ['GroupID', 'PeopleInEachGroup', 'HomePlanet', 'CryoSleep', 'Deck', 'DeckNum',
 'DeckSide', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
 'RoomService_na', 'FoodCourt_na', 'ShoppingMall_na', 'Spa_na', 'VRDeck_na']

    # Delete these features
TO_DELETE = ['PassengerId', 'Cabin', 'Name']

    # Equal Frequency Discretisation
TO_EFD_Q7 = ['Age', 'DeckNum']
TO_EFD_Q9 = ['GroupID']
    
    # Decision Tree Discretisation
TO_DECISION_TREE_DISC = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Weight of Evidence Encoding
TO_WOE = ['HomePlanet', 'CryoSleep', 'VIP', 'DeckSide']

    # One Hot Encoder Top Categories
TO_OHE_TOP_TWO = ['Destination']
TO_OHE_TOP_FIVE = ['Deck']

# Setting up the Pipeline
classification_pipeline = Pipeline([
    
    # Replacing zeros to np.nan in 'Age'
    # ('replace_to_nan', ReplaceZerosWithNaN(value_to_replace = 0, variables = MEAN_IMPUTER)),
    
    # Mean Imputation
    ('mean_imputer', MeanImputer(variables = MEAN_IMPUTER)),
    
    # Adding Binary Indicator
    # ('missing_indicator', AddMissingIndicator(variables = BINARY_AND_MEDIAN_IMPUTER)),
    
    # Median Inputation
    ('median_imputer', MeanMedianImputer(imputation_method = 'median', variables = BINARY_AND_MEDIAN_IMPUTER)),
    
    # Categorical Imputation
    ('categorical_imputation', RandomSampleImputer(random_state = 12, variables = RANDOM_IMPUTER)),
    
    # Additional Features
    ('additional_features', AdditionalFeatures()),
    
    # Drop Features
    ('delete_features', DropFeatures(TO_DELETE)),
    
    # Equal Frequencey Discretistion
    # ('efd_07', EqualFrequencyDiscretiser(q = 7, variables = TO_EFD_Q7)),
    
    # Equal Frequency Discretisation
    # ('efd_09', EqualFrequencyDiscretiser(q = 9, variables = TO_EFD_Q9)),
    
    # Decision Tree Discretisation
    # ('decision_tree_disc', DecisionTreeDiscretiser(cv = 5, scoring = 'roc_auc', 
    #                                                variables = TO_DECISION_TREE_DISC,
    #                                                regression = False, 
    #                                                param_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                                                              'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})),
    
    # Weight of Evidence
    # ('woe', WoEEncoder(variables = TO_WOE)),
    
    # One Hot Encoding
    ('woe', OneHotEncoder(variables = TO_WOE, drop_last = False)),
    
    ('ohe_top_two', OneHotEncoder(variables = TO_OHE_TOP_TWO, drop_last = False)),
    
    # One Hot Encoding
    ('ohe_top_five', OneHotEncoder(variables = TO_OHE_TOP_FIVE, drop_last = False)),
    
    # Standardisation
    ('standard_scaler', MinMaxScaler()),
    
    # Model 
    # ('logit', LogisticRegression(random_state=967, max_iter=200))
    ('logit', RandomForestClassifier(n_estimators=800, random_state=0, max_depth=None, min_samples_split = 2))
    
    
        # ('xgb', xgb.XGBClassifier(random_state=1000, n_estimators = 1049, max_depth = 10, learning_rate = 0.226376,
        #                           booster = 'dart', gamma = 9.688593, subsample = 0.875223, colsample_bytree = 0.583697,
        #                           colsample_bylevel = 0.500000, colsample_bynode = 0.900000))
      
     # ('xgb', xgb.XGBClassifier())
    
    ])

classification_pipeline.fit(X_train, y_train)

# Predictions on train set
pred_train = classification_pipeline.predict_proba(X_train)[:, 1]
print("ROC_AUC for Logistic Regression (TRAIN): ", roc_auc_score(y_train, pred_train))
print("Train Accuracy: ", accuracy_score(y_train, classification_pipeline.predict(X_train)))

# Predictions on test set
pred_test = classification_pipeline.predict_proba(X_test)[:, 1]
print("ROC_AUC for Logistic Regression (TEST): ", roc_auc_score(y_test, pred_test))
print("Test Accuracy: ", accuracy_score(y_test, classification_pipeline.predict(X_test)))

# Confusion matrix
from sklearn.metrics import confusion_matrix

print("For training set: \n", confusion_matrix(y_train, classification_pipeline.predict(X_train)))
print("For testing set: \n", confusion_matrix(y_test, classification_pipeline.predict(X_test)))

# Scoring New Data
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Submitting scores
sample_submission['Transported'] = classification_pipeline.predict(test_data)
sample_submission['Transported'] = sample_submission['Transported'].astype(bool)
sample_submission[['PassengerId', 'Transported']].to_csv('sample_submission.csv', index = False)


"""
test_size = 0.3

For training set: 
 [[2934   77]
 [  49 3025]]
For testing set: 
 [[1069  235]
 [ 355  949]]

ROC_AUC for Logistic Regression (TEST):  0.8415864117957018
Test Accuracy:  0.7737730061349694


For training set: 
 [[2929   83]
 [  47 3026]]
For testing set: 
 [[1058  245]
 [ 317  988]]

ROC_AUC for Logistic Regression (TEST):  0.8655566435252571
Test Accuracy:  0.7845092024539877 


For training set: 
 [[2773  239]
 [ 123 2950]]
For testing set: 
 [[1033  270]
 [ 277 1028]]
 
 ROC_AUC for Logistic Regression (TEST):  0.8817894455177119
 Test Accuracy:  0.790260736196319
 
For training set: 
 [[2759  253]
 [ 174 2899]]
For testing set: 
 [[1028  275]
 [ 266 1039]]

0.79097
ROC_AUC for Logistic Regression (TEST):  0.8824492844393869
Test Accuracy:  0.7925613496932515

0.79494
For training set: 
 [[2481  531]
 [ 492 2581]]
For testing set: 
 [[1017  286]
 [ 228 1077]]
 
"""

"""
New Accuracy: 0.79798
ROC_AUC for Logistic Regression (TRAIN):  0.9178884310896127
Train Accuracy:  0.8274445357436319

ROC_AUC for Logistic Regression (TEST):  0.8918025893678895
Test Accuracy:  0.7963957055214724

For training set: 
 [[2444  568]
 [ 482 2591]]
For testing set: 
 [[ 996  307]
 [ 224 1081]]
 
"""

"""
0.79588

ROC_AUC for Logistic Regression (TRAIN):  0.9346309306650176
Train Accuracy:  0.8525883319638455
ROC_AUC for Logistic Regression (TEST):  0.8972871328469815
Test Accuracy:  0.8098159509202454

For training set: 
 [[2578  434]
 [ 463 2610]]
For testing set: 
 [[1047  256]
 [ 240 1065]]
"""
# --> To Submit
"""
ROC_AUC for Logistic Regression (TRAIN):  1.0
Train Accuracy:  1.0
ROC_AUC for Logistic Regression (TEST):  0.8915604690046324
Test Accuracy:  0.817101226993865
For training set: 
 [[3004    0]
 [   0 3081]]
For testing set: 
 [[1084  227]
 [ 250 1047]]
"""