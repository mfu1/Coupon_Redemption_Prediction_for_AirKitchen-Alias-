
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import mode, entropy
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json
import sys

from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

def main(CV, CRITERION, N_ESTIMATORS, MAX_DEPTH, MAX_FEATURES, BALANCE, RANDOM_STATE, N_JOBS):

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

    # split train & dev
    split_date4 = "2016-05-06"

    trainset4 = trainset[trainset["start_time"] <= split_date4]
    devset4 = trainset[trainset["start_time"] > split_date4]

    # shuffle trainset
    trainset4 = trainset4.iloc[shuffle(trainset4.index).tolist(),]

    X_train = trainset4[trainset4.columns[5:]]
    y_train = trainset4["is_used"]

    X_dev = devset4[devset4.columns[5:]]
    y_dev = devset4["is_used"]

    ## 1. RF

    res_rf = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list)))))
    res_rf["CRITERION"] = CRITERION
    res_rf["BALANCE"] = BALANCE

    evaluations = ["F05", "Precision", "Recall", "Mean_Pre", "AUC", "Accuracy", "OOB"]
    for ev in evaluations:
        for ne in N_ESTIMATORS:
            for md in MAX_DEPTH:
                for mf in MAX_FEATURES:
                    res_rf[ev][str(ne)][str(md)][str(mf)] = []

    # train
    start_time = time.time()

    for ne in N_ESTIMATORS:
        start_time2 = time.time()
        for md in MAX_DEPTH:
            start_time3 = time.time()
            for mf in MAX_FEATURES:
                start_time4 = time.time()

                for n in cv:
                    rf = RandomForestClassifier(n_estimators=ne, max_depth=md, max_features=mf,
                                                criterion = CRITERION,
                                                class_weight={1: BALANCE},
                                                oob_score=True,
                                                random_state=RANDOM_STATE, n_jobs=N_JOBS)
                        
                    rf.fit(X_trains[n], y_trains[n])
                    y_pred = svc.predict(X_devs[n])
                    y_dev = y_devs[n]

                    print("CR:{}, CV: {}, NE: {}, MD: {}, MF: {}".format(CRITERION, n, ne, md, mf))
                    print(confusion_matrix(y_dev, y_pred, labels=[1,0]))
                    
                    f05 = fbeta_score(y_dev, y_pred, beta=0.5, labels=[1,0])
                    precision = precision_score(y_dev, y_pred, labels=[1,0])
                    recall = recall_score(y_dev, y_pred, labels=[1,0])
                    mp = average_precision_score(y_dev, y_pred)
                    auc = roc_auc_score(y_dev, y_pred)
                    acc = accuracy_score(y_dev, y_pred)
                    oob = rf.oob_score_
                    evaluations_res = [f05, precision, recall, mp, auc, acc, oob]
                    
                    for i in range(len(evaluations)):
                        print("{}: {}".format(evaluations[i], evaluations_res[i]))
                        res_rf[evaluations[i]][str(ne)][str(md)][str(mf)].append(evaluations_res[i])
                    print("\n")
                    
                    print("Finished ne {} md {} mf {} in {} sec\n".format(ne, md, mf, time.time() - start_time4))
                    
            print("Finished ne {} md {} in {} sec\n".format(ne, md, time.time() - start_time3))
                
        print("Finished ne {} in {} sec\n".format(ne, time.time() - start_time2))
            
    print("{} sec\n".format(time.time() - start_time))

    # average cv results
    for ev in evaluations:
        for ne in res_rf[ev]:
            for md in res_rf[ev][ne]:
                res_rf[ev][ne][md] = {mf:np.mean(res_rf[ev][ne][md][mf]) for mf in res_rf[ev][ne][md]}

    # save param output
    with open('res_rf_{}_1v{}.json'.format(CRITERION, BALANCE), 'w') as f:
        json.dump(res_rf, f)


if __name__ == "__main__":

    # initialization: CV, CRITERION, BALANCE, N_JOBS
    CV = sys.argv[1] # 2
    if CV == "1":
        cv = [0]
    elif CV == "2":
        cv = [3]
    elif CV == "3":
        cv = [0,1,2,3]
    
    CRITERION = sys.argv[2] # gini or entropy

    BALANCE = int(sys.argv[2]) # 2
    N_JOBS = int(sys.argv[3]) # 4

    N_ESTIMATORS = [10, 50, 100, 300, 500]
    MAX_DEPTH = [2, 3, 4, 6, 8]
    MAX_FEATURES = [4, 6, 8]
    RANDOM_STATE = 42

    print("CRITERION: {}, BALANCE: {}, N_JOBS: {}\n".format(CRITERION, BALANCE, N_JOBS))

    main(CV, CRITERION, N_ESTIMATORS, MAX_DEPTH, MAX_FEATURES, BALANCE, RANDOM_STATE, N_JOBS)

