import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier  
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

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X, Y)

X_predict = missing[feature_columns]
Missing_Value = classifier.predict(X_predict)

missing_column = []
for value in Missing_Value:
    missing_column.append(value)
missing_column = np.array(missing_column)

missing[predicting_column] = missing_column

missing.to_csv('knn/missing_classified.csv',index=False)

training_data.to_csv('knn/removed_missing_values.csv',index=False)

data = training_data.append(missing)

data.to_csv('knn/filled_by_knn.csv',index=False)

columns = range(1,11)
data.iloc[:,columns].to_csv('knn/filled_and_selected_training.csv',index=False)

