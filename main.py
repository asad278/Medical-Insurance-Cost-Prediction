import pandas as pd     # structuring
import numpy as np      # calculations

#data visualization graph
import matplotlib.pyplot as plt
import seaborn as sns

# model, training and testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# import data
insurance_dataset = pd.read_csv('/media/af/HDD/Project/Medical Insurance Cost Prediction/insurance.csv') # data form: age, sex, bmi, children, smoker, region, charges

# print(insurance_dataset.head())    -> show top 5 data of dataframe
# print(insurance_dataset.shape) / print(insurance_dataset.info())   -> detail of dataframe
# print(insurance_dataset.describe())     -> statistic data

# distribution of dataset:
# sns.set()
# sns.displot(insurance_dataset['age'])
# plt.title('Age Distribution')
# plt.show()
# sns.countplot(x='sex', data=insurance_dataset)
# plt.title('Sex Distribution')
# plt.show()

# print(insurance_dataset.isnull().sum())   -> checking for missing data
# correct dataset -> insurance_dataset['<data having null>'].fillna(insurance_dataset['<data having null>'].median(), inplace=True)

# Prepocessing/Encoding
insurance_dataset.replace({'sex': {'male': 1, 'female': 0}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 1, 'no': 0}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
# print(insurance_dataset.head())

# Spiliting features and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
# print(features.head())
# print(target.head())

# Spliting training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(features.shape, features_train.shape, features_test.shape)

# Model training
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediction
train_data_predict = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, train_data_predict)

test_data_predict = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_predict)

# print('Train -> R squared score: ', r2_train)
# print('Test -> R squared score: ', r2_test)

# Predictive System
input_data = (21,0,25.8,0,0,1)
numpy_input = np.asarray(input_data)
final_input = numpy_input.reshape(1, -1)
predict = regressor.predict(final_input)
print(f"Insurance cost is {predict[0]} USD")