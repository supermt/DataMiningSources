import numpy as np
import csv
import pandas as pd
from scipy.stats import chisquare

columns = []
data = []
features = []
headers = []



with open('cleaned-data.csv','r') as myFile:
    lines=csv.reader(myFile)
    headers = next(lines)
    column = 0
    for header in headers:
        # print(header + " " + str(column))
        columns.append(header)
    line_count = 0
    for line in lines:
        # print(line)
        data.append(line)
        features.append([])
        for i in range(1,10):
            if line[i] == "?":
                features[line_count].append(0)
            else:
                features[line_count].append(int(line[i]))
        line_count = line_count + 1

features = np.matrix(features).T

print("Correlation Matrix")
corrcoef_matrix = np.corrcoef(features)
print("Covariance Matrix")
cov_matrix = np.cov(features)
print("Chi-Square Test")


# with open("Chi-Square Test.csv",'w') as f:
#     # f_csv = csv.writer(f)
#     # f_csv.writerows(chisquare(features))

with open('Correlation Matrix.csv','w') as f:
    f_csv = csv.writer(f)
    # f_csv.writerow(headers)
    f_csv.writerows(corrcoef_matrix)

with open('Covariance Matrix.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(cov_matrix)