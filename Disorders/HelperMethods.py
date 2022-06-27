from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TransformInstitueName(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables must be of type: list")
        self.variables = variables
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if len(self.variables) == 1:
            self.variables = self.variables[0]
        
        X['Institution_Applicability_na'] = np.where(X[self.variables] == 'Not applicable', 'Not applicable', np.where(X[self.variables].isna(), np.nan, 'Applicable'))
        X.drop([self.variables], axis = 1)
        X['Institution_Applicability_na'] = np.where(X['Institution_Applicability_na'] == 'nan', 'Missing', X['Institution_Applicability_na'])
        
        return X
            

class ModeImputation(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables must of type: list")
        self.variables = variables
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if len(self.variables) == 1:
            self.variables = self.variables[0]
        
            mode = X[self.variables].mode()[0]
            X[self.variables] = X[self.variables].fillna(mode)
            
            return X
        
        else:
            for feature in self.variables:
                mode = X[feature].mode()[0]
                X[feature] = X[feature].fillna(mode)
                
            return X


class TranformHOSubstanceAbuse(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables must of type: list")
        
        print("Inside Constructor")
        
        self.variables = variables
    
    def fit(self, X, y = None):
        print("Inside Fit")
        return self
    
    def transform(self, X):
        print("Inside Transform")
        X = X.copy()
        print("Inside Transform")
        
        if len(self.variables) == 1:
            print("Inside Transform")
            self.variables = self.variables[0]
        
        print("Inside Transform")
        X[self.variables] = np.where(X[self.variables].isna(), '-', X[self.variables])
        X[self.variables] = np.where(X[self.variables] == '-', 'Missing', X[self.variables])
        
        return X

class FillWithGivenValue(BaseEstimator, TransformerMixin):
    def __init__(self, variables, fill_value):
        if not isinstance(variables, list):
            raise ValueError("Variables must be of type: list")
        if not isinstance(fill_value, str):
            raise ValueError("Fill Value must be of type: str")
        
        self.variables = variables
        self.fill_value = fill_value
            
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if len(self.variables) == 1:
            self.variables = self.variables[0]
        
        X[self.variables] = X[self.variables].fillna(self.fill_value)
        
        return X

class ReplaceExistingValue(BaseEstimator, TransformerMixin):
    def __init__(self, variables, old_value, new_value):
        if not isinstance(variables, list):
            raise ValueError("Variables must be of type: list")
        if not isinstance(old_value, str):
            raise ValueError("Old value must be of type: str")
        if not isinstance(new_value, str):
            raise ValueError("New value must be of type: str")
        
        self.variables = variables
        self.old_value = old_value
        self.new_value = new_value
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if len(self.variables) == 1:
            self.variables = self.variables[0]
            X[self.variables] = np.where(X[self.variables] == self.old_value, self.new_value, self.old_value)
            
            return X
        
        else:
            for feature in self.variables:
                X[feature] = np.where(X[feature] == self.old_value, self.new_value, X[feature])
            
        return X

