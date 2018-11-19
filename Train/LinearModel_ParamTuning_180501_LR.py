
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import mode, entropy
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json
import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

def main(cv, scaler, PENALTY, SOLVER, C, N_JOBS, RANDOM_STATE):

    trainset = pd.read_csv("trainset_180314.csv").iloc[:,1:]
    print(len(trainset))
    print(trainset.columns[5:].tolist())

    # continuous and categorical
    mains = ["user_coupon", "user_id", "coupon_id", "start_time", "is_used"]

    categorical = ['sex_1', 'sex_2', 
                   'age_60', 'age_70', 'age_80', 'age_90', 'age_0', 
                   'city1', 'city2', 'city3', 'city4', 'city5', 
                   'AppVerLast_2.1', 'AppVerLast_2.2', 'AppVerLast_2.3', 'AppVerLast_2.4', 'AppVerLast_2.5', 'AppVerLast_2.7', 'AppVerLast_2.8',
                   'covers_mon', 'covers_tue', 'covers_wed', 'covers_thu', 'covers_fri', 'covers_sat', 'covers_sun', 
                   'type1', 'type6', 
                   'Complaints', 'Eventsoperation', 'NewUserCouponPackageByBD', 'PreUserCouponCode', 'RecallUserDaily', 'home201603222253', 
                   'home_dongbeiguan', 'home_jiangzhecai', 'home_muqinjie', 'home_xiangcaiguan', 'preuser', 'shareuser', 
                   '商家拒单返券', '家厨发券', '活动赠券', '码兑券', '自运营赠券', '蒲公英受邀',
                   'CoupUseLast']

    conitnuous = ['kitchen_entropy', 
                  'distance_median', 'distance_std',
                  'user_longitude_median', 'user_longitude_std', 'user_latitude_median', 'user_latitude_std', 
                  'coupon_effective_days', 'money', 'max_money', 
                  'WeeklyCouponUsedCount', "BiWeeklyCouponUsedCount",
                  'WeeklyOrderCount', 'BiWeeklyOrderCount',
                  'coupon_usage_rate', 'order_coupon_usage_rate',
                  'coupon_type1_usage_rate', 'coupon_type6_usage_rate',
                  'coupon_used_weekend_perc', 'order_weekend_perc', 
                  'worth_money_median', 'worth_money_std', 
                  'InterCoup', 'InterOrder', 'Recency']

    # scaling
    X_train_continuous = scaler.fit_transform(trainset[conitnuous])
    trainset_scaled = pd.concat([trainset.loc[:,mains + categorical], pd.DataFrame(X_train_continuous, columns = conitnuous)], axis=1)

    # split train & dev
    split_date1 = "2016-04-15"
    split_date2 = "2016-04-22"
    split_date3 = "2016-04-29"
    split_date4 = "2016-05-06"

    trainset1 = trainset_scaled[trainset_scaled["start_time"] <= split_date1]
    devset1 = trainset_scaled[(trainset_scaled["start_time"] > split_date1) & (trainset_scaled["start_time"] <= split_date2)]

    trainset2 = trainset_scaled[trainset_scaled["start_time"] <= split_date2]
    devset2 = trainset_scaled[(trainset_scaled["start_time"] > split_date2) & (trainset_scaled["start_time"] <= split_date3)]

    trainset3 = trainset_scaled[trainset_scaled["start_time"] <= split_date3]
    devset3 = trainset_scaled[(trainset_scaled["start_time"] > split_date3) & (trainset_scaled["start_time"] <= split_date4)]

    trainset4 = trainset_scaled[trainset_scaled["start_time"] <= split_date4]
    devset4 = trainset_scaled[trainset_scaled["start_time"] > split_date4]

    # shuffle trainset
    trainset1 = trainset1.iloc[shuffle(trainset1.index).tolist(),]
    trainset2 = trainset2.iloc[shuffle(trainset2.index).tolist(),]
    trainset3 = trainset3.iloc[shuffle(trainset3.index).tolist(),]
    trainset4 = trainset4.iloc[shuffle(trainset4.index).tolist(),]

    trainsets = [trainset1, trainset2, trainset3, trainset4]
    devsets = [devset1, devset2, devset3, devset4]

    X_trains, y_trains, X_devs, y_devs = [], [], [], []
    for i in trainsets:
        X_trains.append(i[i.columns[5:]])
        y_trains.append(i["is_used"])
    for i in devsets:
        X_devs.append(i[i.columns[5:]])
        y_devs.append(i["is_used"])

    ## 1. Logistic Regression

    res_lr = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list)))
    res_lr["PENALTY"] = PENALTY
    res_lr["SCALER"] = SCALER
    res_lr["BALANCE"] = BALANCE

    evaluations = ["F05", "Precision", "Recall", "Mean_Pre", "AUC", "Accuracy"]
    for c in C:
        for ev in evaluations:
            res_lr[ev][str(c)] = []

    # train
    start_time = time.time()

    for c in C:
        start_time2 = time.time()
        
        for n in cv:
            lr = LogisticRegression(C=c,
                                    penalty=PENALTY, solver=SOLVER, 
                                    class_weight={1: BALANCE},
                                    max_iter=MAX_ITER,
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS)
            lr.fit(X_trains[n], y_trains[n])
            y_pred = lr.predict(X_devs[n])
            y_dev = y_devs[n]

            print("P: {}, CV: {}, C: {}".format(PENALTY, n, c))
            print(confusion_matrix(y_dev, y_pred, labels=[1,0]))
            
            f05 = fbeta_score(y_dev, y_pred, beta=0.5, labels=[1,0])
            precision = precision_score(y_dev, y_pred, labels=[1,0])
            recall = recall_score(y_dev, y_pred, labels=[1,0])
            mp = average_precision_score(y_dev, y_pred)
            auc = roc_auc_score(y_dev, y_pred)
            acc = accuracy_score(y_dev, y_pred)
            evaluations_res = [f05, precision, recall, mp, auc, acc]

            for i in range(len(evaluations)):
                print("{}: {}".format(evaluations[i], evaluations_res[i]))
                res_lr[evaluations[i]][str(c)].append(evaluations_res[i])
            print("\n")

        print("Finished c {} in {} sec\n".format(c, time.time() - start_time2))
        
    print("{} sec\n".format(time.time() - start_time))

    # average cv results
    for ev in evaluations:
      res_lr[ev] = {c:np.mean(res_lr[ev][c]) for c in res_lr[ev]}

    # save param output
    with open('res_lr_{}_{}_1v{}.json'.format(PENALTY, SCALER, BALANCE), 'w') as f:
        json.dump(res_lr, f)


if __name__ == "__main__":

    # initialization: CV, SCALER, PENALTY, N_JOBS
    CV = sys.argv[1] # 3
    if CV == "1":
        cv = [0]
    elif CV == "2":
        cv = [3]
    elif CV == "3":
        cv = [0,1,2,3]

    SCALER = sys.argv[2] # 1 or 3
    if SCALER == "1":
        scaler = MinMaxScaler()
    elif SCALER == "2":
        scaler = StandardScaler()
    elif SCALER == "3":
        scaler = MaxAbsScaler()
    elif SCALER == "4":
        scaler = RobustScaler()
        
    PENALTY = sys.argv[3] # l1 or l2
    if PENALTY == "l1":
        SOLVER = "saga"
        MAX_ITER = 1000
        C = [0.01, 0.05, 0.1, 0.5, 1]
    else:
        SOLVER = "sag"
        MAX_ITER = 500
        C = [0.01, 0.1, 1, 10, 100]

    BALANCE = int(sys.argv[4]) # 2
    N_JOBS = int(sys.argv[5]) # 4

    RANDOM_STATE = 42

    print("CV: {}, SCALER: {}, PENALTY: {}, BALANCE: {}, N_JOBS: {}\n".format(cv, SCALER, PENALTY, BALANCE, N_JOBS))

    main(cv, scaler, PENALTY, SOLVER, C, N_JOBS, RANDOM_STATE)

