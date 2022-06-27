import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from sklearn.pipeline import Pipeline
from tensorflow.keras.utils import to_categorical

from feature_engine.imputation import RandomSampleImputer, AddMissingIndicator
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser

from HelperMethods import TransformInstitueName, ModeImputation, TranformHOSubstanceAbuse, FillWithGivenValue, ReplaceExistingValue
from feature_engine.encoding import OrdinalEncoder

from feature_engine.encoding import OneHotEncoder

df1 = pd.read_csv('train_genetic_disorders.csv')
df1 = df1.drop(['Patient Id', 'Patient First Name', 'Family Name', "Father's name"], axis = 1)
df1 = df1.dropna(subset = ['Genetic Disorder'], axis = 0)
df1 = df1.reset_index()

df2 = pd.read_csv('feature_engineering.csv')
df2.drop(['Unnamed: 0'], axis=1, inplace=True)
df = pd.concat([df1, df2], axis = 1)
df = df.drop(['index'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Genetic Disorder'], axis = 1),
                                                    df['Genetic Disorder'],
                                                    test_size=0.3,
                                                    random_state = 24)

# no_of_tests
X_train['no_of_tests'] = X_train[['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']].sum(axis = 1)
X_test['no_of_tests'] = X_test[['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']].sum(axis = 1)

X_train = X_train.drop(['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'], axis = 1)
X_test = X_test.drop(['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'], axis = 1)

# no_of_symptoms
X_train['no_of_symptoms'] = X_train[['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']].sum(axis = 1)
X_test['no_of_symptoms'] = X_test[['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5']].sum(axis = 1)

X_train = X_train.drop(['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'], axis = 1)
X_test = X_test.drop(['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5'], axis = 1)

X_train.columns

# The Target Value
def convert_y(value):
    target_dict = {'Mitochondrial genetic inheritance disorders': 0, 'Single-gene inheritance diseases': 1, 'Multifactorial genetic inheritance disorders': 2}
    
    return target_dict[value]

y_train = y_train.map(convert_y)
y_test = y_test.map(convert_y)

y_cat_train = to_categorical(y_train.values, 3)
y_cat_test = to_categorical(y_test.values, 3)


# Configuration
RANDOM_SAMPLE_IMPUTATION = ['Patient Age', "Mother's age", "Father's age", 'No. of previous abortion', 
                            'White Blood cell count (thousand per microliter)']

EWD_B3 = ['Patient Age']
EWD_B4 = ['No. of previous abortion']
EFD_Q3 = ["Mother's age", "Father's age"]
EFD_Q14 = ['White Blood cell count (thousand per microliter)']

ADD_MISSING_INDICATOR = ['Institute Name']

INSTITUTION_APPLICABILITY = ['Institute Name']
MODE_IMPUTATION = ['Maternal gene']
ADDING_MISSING_INDICATOR_MANUALLY = ['H/O substance abuse']
FILL_YES = ['Inherited from father']
FILL_NONE = ['Autopsy shows birth defect (if applicable)']

REPLACE_EMPTY_WITH_OTHER_CATEOGRY = ['City', 'State', 'Country', 'Zipcode']
ORDINAL_ENCODER = ['Zipcode']

CATEGORICAL_FEATURES_NA = ['Respiratory Rate (breaths/min)', 'Heart Rate (rates/min', 'Parental consent', 
                           'Follow-up', 'Gender', 'Birth asphyxia', 'Place of birth', 
                           'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 
                           'H/O radiation exposure (x-ray)', 'Assisted conception IVF/ART', 
                           'History of anomalies in previous pregnancies', 'Birth defects', 
                           'Blood test result', 'Disorder Subclass']

CATEGORICAL_FEATURES = ["Genes in mother's side", 'Inherited from father', 'Maternal gene', 
                        'Paternal gene', 'Status', 'Respiratory Rate (breaths/min)', 'Heart Rate (rates/min', 
                        'Parental consent', 'Follow-up', 'Gender', 'Birth asphyxia', 
                        'Autopsy shows birth defect (if applicable)', 'Place of birth', 
                        'Folic acid details (peri-conceptional)', 'H/O serious maternal illness', 
                        'H/O radiation exposure (x-ray)', 'H/O substance abuse', 'Assisted conception IVF/ART', 
                        'History of anomalies in previous pregnancies', 'Birth defects', 'Blood test result',
                        'Disorder Subclass', 'City', 'State', 'Country', 'Institution_Applicability_na']

pipe = Pipeline(steps=([
    # Random Sample Imputation
    ('random_imputer', RandomSampleImputer(variables=(RANDOM_SAMPLE_IMPUTATION))),
    
    # Discretisation
    ('ewd_b3', EqualWidthDiscretiser(bins = 3, variables = EWD_B3)),
    ('ewd_b4', EqualWidthDiscretiser(bins = 4, variables = EWD_B4)),
    ('efd_q3', EqualFrequencyDiscretiser(q = 3, variables = EFD_Q3)),
    ('efd_q14', EqualFrequencyDiscretiser(q = 14, variables = EFD_Q14)),
    
    # Add Missing Indicator
    ('missing_indicator_imputation', AddMissingIndicator(variables = ADD_MISSING_INDICATOR)),
    
    # HelperMethods -> TransformInstituteName
        # Introduce column -> Institution_Applicability_na
        # drop 'Institute Name'
    ('institution_applicability', TransformInstitueName(variables = INSTITUTION_APPLICABILITY)),
    
    # HelperMethods -> ModeImputation
    ('mode_imputer', ModeImputation(variables = MODE_IMPUTATION)),
    
    # HelperMethods -> TranformHOSubstanceAbuse
    ('ho_substance', TranformHOSubstanceAbuse(variables = ADDING_MISSING_INDICATOR_MANUALLY)),
    
    # HelperMethods -> FillWithGivenValue
    #('father_inhert', FillWithGivenValue(variables = FILL_YES, fill_value='Yes')),
    #('autopsy', FillWithGivenValue(variables = FILL_NONE, fill_value='None')),
    
    # HelperMethods -> ReplaceExistingValue
    #('replace_values', ReplaceExistingValue(variables = REPLACE_EMPTY_WITH_OTHER_CATEOGRY, old_value='-', new_value='Other')),
    
    # Ordinal Encoding
    #('ordinal_encoder', OrdinalEncoder(encoding_method='arbitrary', variables = ORDINAL_ENCODER)),
    
    # MODE IMPUTATION
    #('mode_imputation_features', ModeImputation(variables = CATEGORICAL_FEATURES_NA)),
    
    # ONEHOT ENCODING
    #('ohe', OneHotEncoder(variables = CATEGORICAL_FEATURES, drop_last=False))
    
    ]))


pipe.fit(X_train, y_train)
X_train = pipe.transform(X_train)






