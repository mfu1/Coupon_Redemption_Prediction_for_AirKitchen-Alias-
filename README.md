# Coupon_Redemption_Prediction_for_AirKitchen-Alias-

SI699 Project
Take the Bait: Predicting Coupon Redemption on a Sharing Economy Site

## Getting Started
This readme provides detailed instructions on how to use data, extract features, train models, and evaluate the system.
 
## Dataset
Folder: ./Dataset/
The data is preprocessed by FeatureEngineering.ipynb and used in ModelSelection.ipynb under the folder Code. 

## Prerequisites
 In order to run the system, you will need the following packages:
 * Python 3 or Python 2 
* Numpy 1.13.1
* Sklearn 0.18
* Scipy 1.0.0
* matplotlib 1.5.1
 
## Feature Engineering
#### Combine 3 raw datasets
Firstly, we processed the raw datasets in Preprocessing.Rmd in R to combine the datasets, organized the data format such as time stamp, and remove unnecessary features.

#### Finalize dataset
Secondly, we used FeatureEngineering.ipynb to finalize the dataset. In this step, we constructed features based on customer lifetime value, transformed categorical data into dummy variables, processed NA values, and calculated features only used the data before the coupon starting date. The output is the trainset and testset. We separate the validationset in the later process.

## Model Selection
### Linear Models
#### Parameter Tuning
ModelSelection_LR and ModelSelection_SVM (.py and .ipynb version)  are the linear models that we tried. We tuned a series hyper parameters including:
SCALER = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler]
CLASS_WEIGHT = [{1:1}, {1:2}, {1:3}]  # BALANCE = [1, 2, 3]

Typically, for Logistic Regression, we further tuned:
C = [0.01, 0.05, 0.1, 0.5, 1] for L1 norm
C = [0.01, 0.1, 1, 10, 100] for L2 norm

For SVM, we further tuned:
C = [0.01, 0.1, 1, 10, 100]
G = [0.01, 0.1, 1, 10]

After using FeatureSelection_LR.py, FeatureSelection_SVM.py file saving the evaluation scores using different parameters for 4-fold cross validations, we used FeatureSelection_LR.ipynb, FeatureSelection_SVM.ipynb to plot the tuning results and use the best parameter to do the final prediction on the testset.

#### Feature Selection
Since SVM does not outperform Logistic Regression, we focused on using Logistic Regression for further analysis. We used ModelSelection_LassoLarsIC.ipynb for L0 norm feature selection, and ModelSelection_LinearRFE.ipynb for RFE feature selection. 

##### Error Analysis
We used ModelSelection_LinearModel_ReducedFeatures.ipynb to do error analysis. More specifically, we grouped the features into different class, and drop each class to see the difference in the evaluation score.

### Random Forest 
The RF_all_scaler.py is a python script created to run random forest models with three different scalers (robust scaler, min_max scaler, and standard scalers) and to search for the best performing model hyperparameters. 
Below is the list of hyperparameters that we use in the search: 
N_ESTIMATORS = [10, 50, 100, 200, 300]
MAX_DEPTH = [4, 6, 8]
MAX_FEATURES = [4, 6, 8]
CLASS_WEIGHT = [{1: 2}, {1: 1}, {1:3}]

We also record the feature importance for each model iteration and saved into a csv file for feature importance analysis. 

### Gradient Boosting 

The XGB_all_scaler.py is a python script created to run gradient boosting models with two different scalers (robust scaler, min_max scaler) and to search for the best performing model hyperparameters. 
Below is the list of hyperparameters that we use in the search: 
N_ESTIMATORS = [10, 50, 100, 300, 500, 1000]
MAX_DEPTH = [4, 6, 8]
LEARNING_RATE = [0.05, 0.1, 0.15, 0.2]

We also record the feature importance for each model iteration and saved into a csv file for feature importance analysis. 
### Feature/Error Analysis
After we identify Gradient Boosting to be the best model, we grouped features into three main categories: coupon_based, user_based, and RFM. Dropping one set of features at a time, the feature set that drops AUC score the most is coupon_based features. To further analyze the individual features in the coupon_based feature set, we also loop through the list of coupon_based features (from most important to least importance, according to feature importance) and drops one at a time to assess how important each individual feature is. 



Activity Coupons and Type 6 coupons are the two features that drops the AUC score and recall the most. Therefore, we took a look at the distribution of these two features in the correct predictions and also in the incorrect predictions. Please refer to the jupyter notebook BestModel_ErrorAnalysis_0419 for the codes. 


