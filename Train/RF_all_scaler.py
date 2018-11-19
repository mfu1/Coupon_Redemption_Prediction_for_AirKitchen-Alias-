# coding: utf-8

from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import sys
import operator
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

N_ESTIMATORS = [10, 50, 100, 200, 300]
MAX_DEPTH = [4, 6, 8]
MAX_FEATURES = [4, 6, 8]
CLASS_WEIGHT = [{1: 2}, {1: 1}]


param_grid = dict(max_features = MAX_FEATURES, max_depth = MAX_DEPTH, n_estimators = N_ESTIMATORS, class_weight = CLASS_WEIGHT )
results_list = list()
parameters = ParameterGrid(param_grid)

######################### Robust Scalder ######################################

trainset4 = pd.read_csv("trainset4_robust_scaled.csv")
devset4 = pd.read_csv("devset4_robust_scaled.csv")

X_train = trainset4[trainset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]
X_devset = devset4[devset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]

y_train = trainset4['is_used']
y_devset = devset4['is_used']


for g in parameters:

    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf.set_params(**g)
    print "parameters: ", clf.get_params(deep=True)

    clf.fit(X_train, y_train)

    y_pred_class = clf.predict(X_devset)

    if clf.classes_[1] == 1:
        y_pred_prob = clf.predict_proba(X_devset)[:, 1]
    else:
        y_pred_prob = clf.predict_proba(X_devset)[:, 0]

    cmatrix = confusion_matrix(y_devset, y_pred_class, labels=[1,0])
    print(cmatrix)

    F5 = fbeta_score(y_devset, y_pred_class, beta=0.5, labels=[1,0])
    print("F0.5: {}".format(F5))
    
    precision = precision_score(y_devset, y_pred_class, labels=[1,0])
    print("Precision: {}".format(precision))
    
    recall = recall_score(y_devset, y_pred_class, labels=[1,0])
    print("Recall: {}".format(recall))
    
    accuracy = accuracy_score(y_devset, y_pred_class)
    print("Accuracy: {}".format(accuracy))

    roc = roc_auc_score(y_devset, y_pred_prob)
    print("Roc: {}".format(roc))

    avg_precision = average_precision_score(y_devset, y_pred_class)
    print ("average_precision_score: {}".format(avg_precision))

    # feats = {}
    # for feature, importance in zip(X_trainset.columns, clf.feature_importances_):
    #     feats[feature] = importance
    # sorted_feats = sorted(feats.items(), key=operator.itemgetter(1), reverse=True)
    
    results = [g, cmatrix, F5, precision, recall, accuracy, roc, avg_precision]

    results_list.append(results)

    if len(results_list) % 5 == 0 : 
        robust_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
        file_name = "robust_rf2_" + str(len(results_list))
        robust_df.to_csv(file_name, index=False, header=True)

final_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
final_df.to_csv("robust_rf_final2.csv", index=False, header=True)


##################################################################################




######################### MinMax Scalder ######################################

results_list = list()

trainset4 = pd.read_csv("trainset4_MinMax_scaled.csv")
devset4 = pd.read_csv("devset4_MinMax_scaled.csv")

X_train = trainset4[trainset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]
X_devset = devset4[devset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]

y_train = trainset4['is_used']
y_devset = devset4['is_used']

for g in parameters:

    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf.set_params(**g)
    print "parameters: ", clf.get_params(deep=True)

    clf.fit(X_train, y_train)

    y_pred_class = clf.predict(X_devset)

    if clf.classes_[1] == 1:
        y_pred_prob = clf.predict_proba(X_devset)[:, 1]
    else:
        y_pred_prob = clf.predict_proba(X_devset)[:, 0]

    cmatrix = confusion_matrix(y_devset, y_pred_class, labels=[1,0])
    print(cmatrix)

    F5 = fbeta_score(y_devset, y_pred_class, beta=0.5, labels=[1,0])
    print("F0.5: {}".format(F5))
    
    precision = precision_score(y_devset, y_pred_class, labels=[1,0])
    print("Precision: {}".format(precision))
    
    recall = recall_score(y_devset, y_pred_class, labels=[1,0])
    print("Recall: {}".format(recall))
    
    accuracy = accuracy_score(y_devset, y_pred_class)
    print("Accuracy: {}".format(accuracy))

    roc = roc_auc_score(y_devset, y_pred_prob)
    print("Roc: {}".format(roc))

    avg_precision = average_precision_score(y_devset, y_pred_class)
    print ("average_precision_score: {}".format(avg_precision))

    # feats = {}
    # for feature, importance in zip(X_trainset.columns, clf.feature_importances_):
    #     feats[feature] = importance
    # sorted_feats = sorted(feats.items(), key=operator.itemgetter(1), reverse=True)
    
    results = [g, cmatrix, F5, precision, recall, accuracy, roc, avg_precision]

    results_list.append(results)

    if len(results_list) % 5 == 0 : 
        robust_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
        file_name = "MinMax_rf2_" + str(len(results_list))
        robust_df.to_csv(file_name, index=False, header=True)

final_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
final_df.to_csv("MinMax_rf_final2.csv", index=False, header=True)


##################################################################################


######################### Standard Scalder ######################################

results_list = list()

trainset4 = pd.read_csv("trainset4_standard_scaled.csv")
devset4 = pd.read_csv("devset4_standard_scaled.csv")

X_train = trainset4[trainset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]
X_devset = devset4[devset4.columns.difference(['user_coupon', 'user_id', 'coupon_id', 'start_time', 'is_used', 'InterCoup', 'InterOrder', 'Recency'])]

y_train = trainset4['is_used']
y_devset = devset4['is_used']


for g in parameters:

    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    clf.set_params(**g)
    print "parameters: ", clf.get_params(deep=True)

    clf.fit(X_train, y_train)

    y_pred_class = clf.predict(X_devset)

    if clf.classes_[1] == 1:
        y_pred_prob = clf.predict_proba(X_devset)[:, 1]
    else:
        y_pred_prob = clf.predict_proba(X_devset)[:, 0]

    cmatrix = confusion_matrix(y_devset, y_pred_class, labels=[1,0])
    print(cmatrix)

    F5 = fbeta_score(y_devset, y_pred_class, beta=0.5, labels=[1,0])
    print("F0.5: {}".format(F5))
    
    precision = precision_score(y_devset, y_pred_class, labels=[1,0])
    print("Precision: {}".format(precision))
    
    recall = recall_score(y_devset, y_pred_class, labels=[1,0])
    print("Recall: {}".format(recall))
    
    accuracy = accuracy_score(y_devset, y_pred_class)
    print("Accuracy: {}".format(accuracy))

    roc = roc_auc_score(y_devset, y_pred_prob)
    print("Roc: {}".format(roc))

    avg_precision = average_precision_score(y_devset, y_pred_class)
    print ("average_precision_score: {}".format(avg_precision))

    # feats = {}
    # for feature, importance in zip(X_trainset.columns, clf.feature_importances_):
    #     feats[feature] = importance
    # sorted_feats = sorted(feats.items(), key=operator.itemgetter(1), reverse=True)
    
    results = [g, cmatrix, F5, precision, recall, accuracy, roc, avg_precision]

    results_list.append(results)

    if len(results_list) % 5 == 0 : 
        robust_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
        file_name = "standard_rf2_" + str(len(results_list))
        robust_df.to_csv(file_name, index=False, header=True)

final_df = pd.DataFrame(results_list, columns = ["params", "cmatrix", "F5", "precision", "recall", "accuracy", "roc", "avg_precision"])
final_df.to_csv("standard_rf_final2.csv", index=False, header=True)

##################################################################################



