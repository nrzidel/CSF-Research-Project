import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import graphviz
from matplotlib.patches import Patch
from sklearn import clone, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import plot_importance
from CSFData import getter

loop_params = False


# data = pd.read_csv("Data\PPMI_Cohort_Filtered.csv")
# data = data.drop(data.columns[0], axis=1)   #Drop the first column, which is the sequential numbers from R

# y = data["PPMI_COHORT"].values
# X = data.iloc[:, 3:]

# le = LabelEncoder()
# y = le.fit_transform(y)

# data = Process_Data(X,y)
# data.cleanData()
# data.featureSelector(0.9,50)
# data = getter()
# X, y = data.getXy()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# estimators = [
#     # ('encoder', TargetEncoder()),
#     ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
# ]
# pipe = Pipeline(steps=estimators)

# search_space = {
#     'clf__booster': ("gbtree", "dart", "gblinear"),
#     'clf__min_child_weight': Real(0.0, 10),
#     'clf__max_delta_step': Integer(0, 10),
#     'clf__subsample': Real(0, 1),
#     'clf__max_depth': Integer(2,8),
#     'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
#     'clf__subsample': Real(0.5, 1.0),
#     'clf__colsample_bytree': Real(0.5, 1.0),
#     'clf__colsample_bylevel': Real(0.5, 1.0),
#     'clf__colsample_bynode' : Real(0.5, 1.0),
#     'clf__reg_alpha': Real(0.0, 10.0),
#     'clf__reg_lambda': Real(0.0, 10.0),
#     'clf__gamma': Real(0.0, 10.0),
#     # 'clf__tree_method': ("exact", "approx", "hist"),
#     'clf__max_leaves': Integer(0, 10),
#     'clf__num_parallel_tree': Integer(1,5)

# }

# opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=15, scoring='roc_auc', random_state=42) 
# # in reality, you may consider setting cv and n_iter to higher values
# opt.fit(X_train, y_train)
# print(opt.best_estimator_)
# print(opt.best_score_)
# print(opt.score(X_test, y_test))
# opt.predict(X_test)
# opt.predict_proba(X_test)
# opt.best_estimator_.steps
# xgboost_step = opt.best_estimator_.steps[0]
# xgboost_model = xgboost_step[1]
# plot_importance(xgboost_model)

# (base_score=None, booster=None, callbacks=None,   
#                                colsample_bylevel=0.9777389931549642,
#                                colsample_bynode=0.8503107223106829,
#                                colsample_bytree=0.9358259642316076, device=None,
#                                early_stopping_rounds=None,
#                                enable_categorical=False, eval_metric=None,
#                                feature_types=None, feature_weights=None,
#                                gamma=4.158210875794536, grow_policy=None,
#                                importance_type=None,
#                                interaction_constraints=None,
#                                learning_rate=0.01390574606467376, max_bin=None,
#                                max_cat_threshold=None, max_cat_to_onehot=None,
#                                max_delta_step=None, max_depth=3,
#                                max_leaves=None, min_child_weight=None,
#                                missing=nan, monotone_constraints=None,
#                                multi_strategy=None, n_estimators=None,
#                                n_jobs=None, num_parallel_tree=None, ...))])


estimators = [
    # ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)

search_space = {
    # 'clf__booster': ("dart", "gblinear"),
    'clf__min_child_weight': Real(0.0, 10),
    'clf__max_delta_step': Integer(0, 10),
    'clf__subsample': Real(0, 1),
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0),
    # 'clf__tree_method': ("exact", "approx", "hist"),
    'clf__max_leaves': Integer(0, 10),
    'clf__num_parallel_tree': Integer(1,5)

}

opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=15, scoring='roc_auc', random_state=42) 

feature_dictionary = {}

if loop_params:
    highest = 0
    best_model = None
    best_inputs = None
    num_runs = 125
    current_run = 0
    num_failures = 0
    for sheet in [1]:
        for thresh in [0.5]:
            for knn in [11,13,15,17,19]:
                for varthresh in [0.6,0.7,0.8,0.9,1,]:
                    for kselect in [15,17,19,21,23,25]:

                        data = getter(datasheet=sheet)
                        X, y = data.getXy(nathresh=thresh, knn = knn, varthresh=varthresh, kselect=kselect)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                        opt.fit(X_train, y_train)
                        
                        # in reality, you may consider setting cv and n_iter to higher values
                        # opt.fit(X_train, y_train)
                        # print(opt.best_estimator_)
                        # print(opt.best_score_)
                        # print(opt.score(X_test, y_test))
                        # opt.predict(X_test)
                        # opt.predict_proba(X_test)
                        # opt.best_estimator_.steps
                        
                        xgboost_step = opt.best_estimator_.steps[0]
                        xgboost_model = xgboost_step[1]
                        # plot_importance(xgboost_model)
                        if opt.best_score_ > highest:
                            highest = opt.best_score_
                            print("Better Score Found: {}.".format(highest))
                            best_model = opt
                            best_inputs = [sheet, thresh, knn, varthresh, kselect]
                        importances = opt.best_estimator_.steps[0][1].feature_importances_
                        named_importances = list(zip(data.get_X_columns(), importances))

                        for i in range(10):
                            if named_importances[i][0] in feature_dictionary.keys():
                                feature_dictionary[named_importances[i][0]]+=1
                            else:
                                feature_dictionary[named_importances[i][0]]=1
                        
                        current_run+=1
                        print("Run {} of {}".format(current_run, num_runs))
                            
    print(best_model.best_estimator_)
    print(best_model.best_score_)
    print(best_model.score(X_test, y_test))
    print(best_inputs)
    print(num_failures)
    sorted_features = sorted(feature_dictionary, key=lambda x: x[1], reverse=True)
    print(sorted_features)

else:

    features = {'501': 115, '100000295': 60, '100000894': 150, '100001178': 150, '100001315': 150, '100001605': 130, '100002613': 42, '100002927': 59, '100004208': 48, '100004322': 13, '100001403': 86, '100001992': 98, '100001416': 121, '41': 53, '849': 85, '100006191': 1, '100001108': 52, '100004634': 4, '100004284': 15, '100001405': 12, '881': 2, '100001167': 1, '100001538': 23, '100001054': 29, '55': 1}
    sorted_features = sorted(features.items(), key=lambda item: item[1], reverse = True)
    top_15_features = sorted_features[:15]
    feature_list = []
    for item in top_15_features:
        feature_list.append(item[0])

    data = getter(datasheet=1)
    X, y = data.getXy_selectfeatures(columns=feature_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    opt.fit(X_train, y_train)
    
    # in reality, you may consider setting cv and n_iter to higher values

    xgboost_step = opt.best_estimator_.steps[0]
    xgboost_model = xgboost_step[1]
  
    print(opt.best_estimator_)
    print(opt.best_score_)
    print(opt.score(X_test, y_test))

    data2 = getter(datasheet=1, group="V06")
    X2, y2 = data2.getXy_selectfeatures(columns=data.get_X_columns())
    print(opt.score(X2, y2))
    importances = opt.best_estimator_.steps[0][1].feature_importances_
    named_importances = zip(data.get_X_columns(), importances)
    sorted_feature_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_feature_importances:
        print(f"{feature}: {importance}")
    
    # sorted_features = {'501': 115, '100000295': 60, '100000894': 150, '100001178': 150, '100001315': 150, '100001605': 130, '100002613': 42, '100002927': 59, '100004208': 48, '100004322': 13, '100001403': 86, '100001992': 98, '100001416': 121, '41': 53, '849': 85, '100006191': 1, '100001108': 52, '100004634': 4, '100004284': 15, '100001405': 12, '881': 2, '100001167': 1, '100001538': 23, '100001054': 29, '55': 1}