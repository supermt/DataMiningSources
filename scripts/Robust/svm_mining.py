#! /usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
from sklearn import svm
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import statistics 
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import os
from scipy.io import arff
from sklearn.model_selection import train_test_split

KERNEL='poly'

def svm_processing_and_evaluation(data_train,data_test,column_lists=range(1,10)):

    feature_columns = range(0,9)

    predicting_column = "class"

    X_train = data_train.iloc[:,feature_columns]
    X_test = data_test.iloc[:,feature_columns]
    y_train = data_train[predicting_column]
    y_test = data_test[predicting_column]

    TPs=[]
    FNs=[]
    FPs=[]
    TNs=[]
    
    acs = []
    prs = []
    res = []
    sps = []

    y_tests = []
    y_scores = []



    y_tests.extend(y_test)
    clf = svm.SVC(kernel=KERNEL)

    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    y_scores.extend(y_score)

    y_result = clf.predict(X_test)


    TP = 0 # Predict Positive, Actual Postive
    FN = 0 # Predict Negative, Actual Postive
    FP = 0 # Predict Positive, Actual Negative
    TN = 0 # Predict Negative, Actual Negative

    POSTIVE = 4
    NEGATIVE = 2

    for (y_p,y_a) in zip(y_result,y_test):
        # print(y_p,y_a)
        if y_p == POSTIVE and y_a == POSTIVE:
            TP += 1
        elif y_p == NEGATIVE and y_a == POSTIVE:
            FN += 1
        elif y_p == POSTIVE and y_a == NEGATIVE:
            FP += 1
        elif y_p == NEGATIVE and y_a == NEGATIVE:
            TN += 1

    TPs.append(TP)
    FNs.append(FN)
    FPs.append(FP)
    TNs.append(TN)
    TP = float(TP)
    FN = float(FN)
    FP = float(FP)
    TN = float(TN)
    total = TP + FN + FP + TN
    accuracy = (TP + TN) / (total)
    precesion = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score, pos_label=4)
    roc_auc = auc(fpr, tpr)
    plt.figure("ROC of SVM")
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve ( aka AUC = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC of SVM')
    # plt.legend(loc="lower right")
    # plt.show()
    
    # print("Weights assigned to the features are: ",clf.coef_)
    # print("The Constants in decision function: ",clf.intercept_)
    # print("margin distance equals to: ",(1 / np.sqrt(np.sum(clf.coef_ ** 2))))

    print(accuracy,precesion,recall,specificity)
    return (np.sum(TPs),np.sum(FNs),np.sum(FPs),np.sum(TNs))

# # print(clf.predict(X_test))

# np.random.seed(0)
# svd = TruncatedSVD(n_components=2)

# X = svd.fit_transform(X_train)

# # figure number
# fignum = 1

# # fit the model
# clf = svm.SVC(kernel='linear', C=1)
# Y=y_train
# clf.fit(X, Y)

# # get the separating hyperplane
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(2, 20)
# yy = a * xx - (clf.intercept_[0]) / w[1]

# # plot the parallels to the separating hyperplane that pass through the
# # support vectors (margin away from hyperplane in direction
# # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
# # 2-d.
# margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
# yy_down = yy - np.sqrt(1 + a ** 2) * margin
# yy_up = yy + np.sqrt(1 + a ** 2) * margin

# # plot the line, the points, and the nearest vectors to the plane
# plt.clf()
# plt.plot(xx, yy, 'k-')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#             facecolors='none', zorder=10, edgecolors='k')
# plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
#             edgecolors='k')

# plt.axis('tight')
# x_min = 2
# x_max = 20
# y_min = -10
# y_max = 10

# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(XX.shape)

# plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# plt.xticks(())
# plt.yticks(())
# fignum = fignum + 1

# plt.show()

def spliting_prediction(data,column_lists=range(1,10)):

    feature_columns = range(0,9)

    predicting_column = 9

    y = data.iloc[:,predicting_column]

    X = data.iloc[:,feature_columns]

    # print(y)

    X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    TPs=[]
    FNs=[]
    FPs=[]
    TNs=[]

    y_tests = []
    y_scores = []



    y_tests.extend(y_test)
    clf = svm.SVC(kernel=KERNEL)

    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    print("Weights assigned to the features are: ",clf.coef_)
    print("The Constants in decision function: ",clf.intercept_)
    print("margin distance equals to: ",(1 / np.sqrt(np.sum(clf.coef_ ** 2))))

    y_scores.extend(y_score)

    y_result = clf.predict(X_test)


    TP = 0 # Predict Positive, Actual Postive
    FN = 0 # Predict Negative, Actual Postive
    FP = 0 # Predict Positive, Actual Negative
    TN = 0 # Predict Negative, Actual Negative

    POSTIVE = 4
    NEGATIVE = 2

    for (y_p,y_a) in zip(y_result,y_test):
        # print(y_p,y_a)
        if y_p == POSTIVE and y_a == POSTIVE:
            TP += 1
        elif y_p == NEGATIVE and y_a == POSTIVE:
            FN += 1
        elif y_p == POSTIVE and y_a == NEGATIVE:
            FP += 1
        elif y_p == NEGATIVE and y_a == NEGATIVE:
            TN += 1

    TPs.append(TP)
    FNs.append(FN)
    FPs.append(FP)
    TNs.append(TN)
    TP = float(TP)
    FN = float(FN)
    FP = float(FP)
    TN = float(TN)
    total = TP + FN + FP + TN
    accuracy = (TP + TN) / (total)
    precesion = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score, pos_label=4)
    roc_auc = auc(fpr, tpr)
    plt.figure("ROC of SVM")
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve ( aka AUC = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC of SVM')
    # plt.legend(loc="lower right")
    # plt.show()
    # print("accuracy,precesion,recall,specificity")
    # print(accuracy,precesion,recall,specificity)
    return (np.sum(TPs),np.sum(FNs),np.sum(FPs),np.sum(TNs))



if __name__ == "__main__":
    # path = './noise'

    # files = []
    # # r=root, d=directories, f = files
    # for r, d, f in os.walk(path):
    #     for file in f:
    #         if '.arff' in file:
    #             files.append(os.path.join(r, file))
    
    # files.sort()
    
    # for f in files:
    #     print(f)
    #     data = arff.loadarff(f)
    #     df = pd.DataFrame(data[0])
    #     spliting_prediction(df)

    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    # data_train = pd.read_csv("train_no_noise.csv")
    # data_test = pd.read_csv("test.csv")
    
    print(svm_processing_and_evaluation(data_train,data_test))
    # # print("using file without possible outliers: ")
    # # data = pd.read_csv("outliers/training.csv")
    # # print(svm_processing_and_evaluation(data))
    # # plt.figure()
    