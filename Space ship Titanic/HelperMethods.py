import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin

# Importing the train dataset
dataset = pd.read_csv('train.csv')

# Seperating the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(['Transported'], axis=1),
    dataset['Transported'],
    test_size = 0.3,
    random_state = 12
    )

class ReplaceZerosWithNaN(BaseEstimator, TransformerMixin):
    
    def __init__(self, value_to_replace, variables = None):
        
        if not isinstance(variables, list):
            raise ValueError("Variables must be of type: list")
        
        self.value_to_replace = value_to_replace
        self.variables = variables
   
    def fit(self, dataset, y = None):
        if self.variables:
            return self
        
    def transform(self, dataset):
        
        dataset = dataset.copy()
        
        if self.variables:
            dataset[self.variables] = dataset[self.variables].replace(self.value_to_replace, np.nan)
                
        return dataset

class MeanImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables = None):
        
        if not isinstance(variables, list):
            raise ValueError("Variables must be of type: list")
        
        self.variables = variables
    
    def fit(self, dataset, y = None):
        
        self.imputer_dict_ = dict()
        
        for feature in self.variables:
            self.imputer_dict_[feature] = int(X_train[feature].mean())
        
        return self
    
    def transform(self, dataset):
        dataset = dataset.copy()
        
        for feature in self.variables:
            dataset[feature].fillna(self.imputer_dict_[feature], inplace = True)
        
        return dataset

"""
class AdditionalFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, dataset, y = None):
        
        return self
    
    def transform(self, dataset):
        
        dataset = dataset.copy()
        
        dataset.insert(1, 'GroupID', dataset['PassengerId'].apply(lambda x : int(x.split('_')[0])))
        dataset.insert(2, 'PeopleInEachGroup', dataset['PassengerId'].apply(lambda x : int(x.split('_')[1])))
        
        dataset.insert(6, 'Deck', dataset['Cabin'].apply(lambda x: str(x).split('/')[0]))
        dataset.insert(7, 'DeckNum', dataset['Cabin'].apply(lambda x: int(x.split('/')[1])))
        dataset.insert(8, 'DeckSide', dataset['Cabin'].apply(lambda x: str(x).split('/')[2]))
        
        return dataset
"""
























