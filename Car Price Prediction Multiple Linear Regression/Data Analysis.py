import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from scipy.stats import yeojohnson
from scipy import stats

# Importing the dataset
dataset = pd.read_csv('CarPrice_Assignment.csv')

# Default Distribution of the "Target" variable
sns.displot(dataset['price'], kind='kde').set(title = 'Default distribution of "Price"')

# Applying transformations to the "Target" Variable

    # 1. Boxcox Transformation
fitted_data_boxcox, lmda = boxcox(dataset['price'])
sns.displot(fitted_data_boxcox, kind='kde').set(title = "Boxcox Transformation")
sns.rugplot(data = fitted_data_boxcox)

stats.probplot(fitted_data_boxcox, dist="norm", plot=plt)

# print(fitted_data)

        #-> *** Converting box-cox transformed price values to actual price values
actual_values = np.exp(np.log(lmda * fitted_data_boxcox + 1) / lmda)
print(actual_values)

    # 2. Inverse Transformation
sns.displot(1/dataset['price'], kind='kde').set(title = "Inverse Transformation")
sns.rugplot(1/dataset['price'])

stats.probplot(1/dataset['price'], dist='norm', plot=plt)

# Comparing Boxcox and Inverse Transformations 
plt.subplot(1, 2, 1)
stats.probplot(fitted_data_boxcox, dist='norm', plot=plt)
plt.title('Boxcox Transformation')
plt.subplot(1, 2, 2)
stats.probplot(1/dataset['price'], dist='norm', plot=plt)
plt.title('Inverse Transformation')
plt.subplots_adjust(left=0.1, right=1.9)
plt.show()

# Variable Types (Numerical and Categorical)
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'object']
numerical_features = [feature for feature in dataset.columns if feature not in categorical_features and feature != 'price']

# Missing data
print(dataset.isnull().any().sum())

# Numerical variables and it's relation with price 
    # Discrete Variables -> Find Transformations
    # Continous Variables -> Find Transformations

numerical_features.remove('car_ID')

for feature in numerical_features:
    print(feature, "\t", dataset[feature].nunique())

# symboling means every car is assigned with a risk factor symbol with it's price.
# More risky moves up (upto 3) and safe moves down (-3) 
sns.barplot(data = dataset['symboling'], x=dataset['symboling'], y=dataset['price']).set(title = 'Symboling (vs) Price')
plt.axhline(y=17500, color='g')
plt.axhline(y=20500, color='r')
plt.show()

# Wheelbase means the distance between the centres of the front and rear wheels.
# Car with long wheelbase will tend to have a more spacious  interior than a car of the
# same length with a shorter wheelbase.
each_wheelbase_price = []
for wheelbase_value in sorted(dataset['wheelbase'].unique()):
    each_wheelbase_price.append(sum(dataset[dataset['wheelbase'] == wheelbase_value]['price'])) 

sns.lineplot(x=dataset['wheelbase'], y=dataset['price'], markers='x').set(title = 'Wheelbase (vs) Price')
sns.scatterplot(x=each_wheelbase_price, y=dataset['price']) # Error

plt.figure(figsize=(12, 8))
sns.countplot(x=dataset['wheelbase'])
plt.xticks(rotation = 90)
plt.yticks(np.arange(min(dataset['wheelbase'].value_counts()), max(dataset['wheelbase'].value_counts()) + 1, 1))
plt.ylim(ymin=min(dataset['wheelbase'].value_counts()), ymax=max(dataset['wheelbase'].value_counts()))
plt.show()

# Car length
sns.scatterplot(data=dataset, x='carlength', y='price', hue='price', size='price').set(title='Carlength (vs) Price')

# Car Width
sns.scatterplot(data=dataset, x='carwidth', y='price', hue='price', size='price').set(title = 'Carwidth (vs) Price')

# Car Height
sns.scatterplot(data=dataset, x='carheight', y='price', hue='price', size='price').set(title = 'Carheight (vs) Price')

# Curb Weight means the weight of the vehicle minus the passengers, luggage, and accessories and 
# what remains is the standard fitment that comes with from the manufacturer. This includes a full 
# tank of fuel and is measured when the vehicle is not being used and resting on the curb(flat surface).
sns.scatterplot(data=dataset, x='curbweight', y='price', hue='price', size='price').set(title = 'CurbWeight (vs) Price')

# Engine Size is the volume of the fuel and air that can be pushed through a car's cylinders and 
# is measured in cubic centimeters (cc)
sns.scatterplot(data=dataset, x='enginesize', y='price', hue='price', size='price').set(title = 'EngineSize (vs) Price')

# Bore ratio means the ratio between cylinder bore diameter and piston stroke length. Bore is inner
# diameter of the cylinder.
# Bore-Stroke ratio is the ratio between the dimensions of the engine cylinder bore diameter to its
# piston stroke-length. This determines engine power and torque characteristics.
sns.scatterplot(data=dataset, x='boreratio', y='price', hue='price', size='price').set(title = 'BoreRatio (vs) Price')

# Stroke means a phase of the engine's cycle, during which the piston travels from top to bottom or
# vice versa. Stroke length is the distance travelled by the piston during each cycle.
# 4 stroke engine makes two complete revolutions to complete one power stroke where as two stroke
# engine make one complete revolution to complete one power stroke.
sns.scatterplot(data=dataset, x='stroke', y='price', hue='price', size='price').set(title = 'Stroke (vs) Price')

# Compression ratio is defined as the ratio of the volume of the cylinder and its head space.
sns.scatterplot(data=dataset, x='compressionratio', y='price').set(title = 'CompressionRatio (vs) Price')


# Horsepower refers to power an engine produces.
sns.scatterplot(data=dataset, x='horsepower', y='price', hue='price', size='price').set(title = 'Horsepower (vs) Price')

# peak rpm means how fast a machine is running at a particular instance
sns.scatterplot(data=dataset, x='peakrpm', y='price', hue='price', size='price').set(title = 'Peak rpm (vs) Price')

# City mpg is generally the lowest mpg rating for a vehicle primarily because of the frequent 
# starting, stoping and idling.
sns.scatterplot(data=dataset, x='citympg', y='price', hue='price', size='price').set(title = 'City mpg (vs) Price')


# Highway mpg is generally the highest mpg rating because uninterrupted driving tends to burn less fuel
sns.scatterplot(data=dataset, x='highwaympg', y='price', hue='price', size='price').set(title = 'Highway mpg (vs) Price')

# Heat map
sns.heatmap(dataset.corr(), annot=True, linewidths=.5)
sns.set(rc = {'figure.figsize':(10,8)})
plt.show()

# Numerical Transformations
numerical_discrete_features = [feature for feature in numerical_features if dataset[feature].nunique() < 15]
numerical_continous_features = [feature for feature in numerical_features if feature not in numerical_discrete_features]

dataset[numerical_continous_features].hist(figsize=(15, 15))
plt.show()

extremely_skewed_features = ['compressionratio']
skewed_features = [feature for feature in numerical_continous_features if feature not in extremely_skewed_features]

# Yeo-johnson Transformation
temp_data = dataset.copy(deep=True)

for feature in skewed_features:
    temp_data[feature], param = stats.yeojohnson(dataset[feature])

temp_data[skewed_features].hist(figsize=(15, 15))
plt.show()

# Features to transform to Yeojohnson Transformations
# Try: wheelbase, carwidth, enginesize, boreratio
to_yeojohnson_transformation = ['carlength', 'carheight', 'curbweight', 'stroke', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# Probability Plot before and after transformation
for feature in skewed_features:
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    stats.probplot(dataset[feature], dist='norm', plot=plt)
    plt.ylabel('Price')
    plt.xlabel('Original: ' + feature)
    
    plt.subplot(1, 2, 2)
    stats.probplot(temp_data[feature], dist='norm', plot=plt)
    plt.ylabel('Price')
    plt.xlabel('Transformed: ' + feature)
    
    plt.show()

# Scatterplot before and after transformations
for feature in skewed_features:
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(dataset[feature], fitted_data_boxcox)
    plt.ylabel('Price')
    plt.xlabel('Original: ' + feature)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(temp_data[feature], fitted_data_boxcox)
    plt.ylabel('Price')
    plt.xlabel('Transformed: ' + feature)
    
    plt.show()

to_sqrt_transformation = ['boreratio']
to_inverse_transformation = ['enginesize']
to_log_transformation = ['wheelbase', 'carwidth']

# remaining_features = []

# for feature in skewed_features:
#     if feature not in to_yeojohnson_transformation and feature not in to_sqrt_transformation and feature not in to_inverse_transformation:
#         remaining_features.append(feature)

# # Try: wheelbase, carwidth, enginesize, boreratio
# temp_data = dataset.copy(deep = True)

# for feature in remaining_features:
#     temp_data[feature] = np.log(dataset[feature])

# temp_data[remaining_features].hist(figsize=(15, 15))

# # Probability Plot before and after transformation
# for feature in remaining_features:
    
#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 2, 1)
#     stats.probplot(dataset[feature], dist='norm', plot=plt)
#     plt.ylabel('Price')
#     plt.xlabel('Original: ' + feature)
    
#     plt.subplot(1, 2, 2)
#     stats.probplot(temp_data[feature], dist='norm', plot=plt)
#     plt.ylabel('Price')
#     plt.xlabel('Transformed: ' + feature)
    
#     plt.show()

# Binary Transformation
to_binary_transformation = ['compressionratio']

temp_data = dataset.copy(deep=True)

temp_data[to_binary_transformation] = np.where(dataset[to_binary_transformation] > 9, 0, 1)

temp_data = temp_data.groupby(to_binary_transformation)['price'].agg(['mean', 'std'])
temp_data.plot(kind='barh', y='mean', legend=False, xerr='std', title='price', color='green')

# Categorical data
    # Cardinality -> How many different categories present in each categorical variable
    # Quality Features -> If a categorical feature has any quality variables, replace it with numbers. Ex: excellent 3, good 2, better 1 
    # Rare Labels -> Check for categories in a feature with small percentage

# Symboling is Categorical Feature
dataset['symboling'] = dataset['symboling'].astype('object')
categorical_features.append('symboling')

# How many different categories are present in each categorical feature
dataset[categorical_features].nunique().sort_values(ascending=False).plot.bar(figsize=(12, 5))

car_company_names = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 
                     'isuzu', 'jaguar', 'mazda', 'buick', 'mercury cougar', 
                     'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 
                     'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'vw', 
                     'volvo']

company_names_test = []

for i in range(0, len(dataset)):
    for car_company in car_company_names:
        if car_company in dataset['CarName'][i].lower():
            dataset['CarName'].loc[i] = car_company
            print(dataset['CarName'][i], '->', car_company)

# Category with less than 1%
def analyse_rare_labels(data, feature, rare_percentage):
    df = data.copy(deep=True)
    
    temp_data = df.groupby(feature)['price'].count() / len(df)
    
    return temp_data[temp_data < rare_percentage]
    
for feature in categorical_features:
    print(feature, ': ', analyse_rare_labels(dataset, feature, 0.01))
    print("=========================================================")

# Categorial values (vs) Price
for feature in categorical_features:
    sns.catplot(x = feature, y = 'price', data=dataset, kind = 'box', height = 4, aspect = 1.5)
    
    plt.show()

for feature in categorical_features:
    sns.stripplot(data = dataset, x = feature, y = 'price', color = 'k', alpha = 0.3)
    sns.boxenplot(data = dataset, x = feature, y = 'price')
    
    plt.show()




