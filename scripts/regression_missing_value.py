import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('origin.csv')

missing = []

feature_columns = ["Uniformity of Cell Shape","Marginal Adhesion","Bland Chromatin"]
predicting_column = "Bare Nuclei"
# sns.pairplot(data, x_vars=['Uniformity of Cell Shape','Marginal Adhesion','Bland Chromatin'], y_vars='Bare Nuclei', height=7, aspect=0.8)
missing = data.loc[data[predicting_column] == "?"]
training_data = data.loc[data[predicting_column] != "?"]
X = training_data[feature_columns]
Y = training_data[predicting_column]

# X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg = LinearRegression()
model=linreg.fit(X, Y)

X_predict = missing[feature_columns]
Missing_Value = linreg.predict(X_predict)

missing_column = []
for value in Missing_Value:
    value = int(round(value))
    missing_column.append(value)
missing_column = np.array(missing_column)

missing[predicting_column] = missing_column

missing.to_csv('missing_regressioned.csv',index=False)

training_data.to_csv('removed_missing_values.csv',index=False)

data = training_data.append(missing)

data.to_csv('filled_by_regression.csv',index=False)

columns = range(1,11)
data.iloc[:,columns].to_csv('filled_and_selected_training.csv',index=False)
