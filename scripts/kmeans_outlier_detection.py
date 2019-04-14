import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv('knn/filled_by_knn.csv')

feature_columns = range(1,10)
predicting_column = "Bare Nuclei"
# sns.pairplot(data, x_vars=['Uniformity of Cell Shape','Marginal Adhesion','Bland Chromatin'], y_vars='Bare Nuclei', height=7, aspect=0.8)
# missing = data.loc[data[predicting_column] == "?"]
print(data.shape)
# training_data = data.loc[data[predicting_column] != "?"]
X = data.iloc[:,feature_columns]

# X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# print(kmeans.cluster_centers_[kmeans.labels_[0]])
# print(kmeans.labels_[0])

result_dict = {}
index = 0
for index_ori, row in X.iterrows():
    center = kmeans.cluster_centers_[kmeans.labels_[index]]
    points = row.astype(np.float)
    dist = np.linalg.norm(center - points)
    index = index + 1
    result_dict[index_ori] = dist


result_sorted_list = sorted(result_dict.items(),key = lambda item:item[1],reverse=True)
outliers_count = int(0.1 * X.shape[0])
outliers_list = result_sorted_list[0:outliers_count]
# print(result_sorted_list)
index_list = []

for outlier in outliers_list:
    index_list.append(outlier[0])

outliers = data.loc[index_list]

remaining = data.drop(index=index_list)

outliers.to_csv('outliers/outliers.csv',index=False)

remaining.to_csv('outliers/training.csv',index=False)

data.to_csv('outliers/origin.csv',index=False)

feature_columns = range(1,11)
remaining.iloc[:,feature_columns].to_csv('outliers/training_feature_selected.csv',index=False)
data.iloc[:,feature_columns].to_csv('outliers/origin_feature_selected.csv',index=False)
outliers.iloc[:,feature_columns].to_csv('outliers/noises_feature_selected.csv',index=False)



# print(remaining.shape)
# print(data.shape)
# print(outliers.shape)