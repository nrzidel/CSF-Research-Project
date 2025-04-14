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
import pickle

loop_params =  False

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

best_models = []

if loop_params:

    pickle_name = "xgboost_sheet_1"
    num_runs = 100
    current_run = 0
    for sheet in [1]:
        for thresh in [0.5]:
            for knn in [5,10,15,20]:
                for varthresh in [0.6,0.7,0.8,0.9,1,]:
                    for kselect in [20,25,30,40,50]:

                        opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=25, scoring='roc_auc', random_state=42) 
                        data = getter(datasheet=sheet)
                        X, y = data.getXy(nathresh=thresh, knn = knn, varthresh=varthresh, kselect=kselect)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                        opt.fit(X_train, y_train)

                        best_models.append((opt.best_score_, opt, (sheet, thresh, knn, varthresh, kselect), data.get_X_columns()))
                        best_models = sorted(best_models, key=lambda x: x[0], reverse=True)
                        if len(best_models)>10:
                            best_models = best_models[:10]
                        
                        current_run+=1
                        print("Run {} of {}".format(current_run, num_runs))
    
    with open(pickle_name, 'wb') as file:
        pickle.dump(best_models, file)
                            
    best_model = best_models[0][1]
    print(best_model.best_estimator_)
    print(best_model.best_score_)
    print(best_model.score(X_test, y_test))
    print(best_models[0][2])

else:

    pickle_name = "xgboost_sheet_1"
    with open(pickle_name, 'rb') as file:
        best_models = pickle.load(file)

    feature_dictionary = {}
    for model in best_models:
        model_obj = model[1]
        x_columns = model[3]
        importances = model_obj.best_estimator_.steps[0][1].feature_importances_
        named_importances = list(zip(x_columns, importances))
        print("Best_Score: {}".format(model[0]))
        print(model_obj.best_estimator_)    
        print(named_importances)
        for i in range(20):
                if named_importances[i][0] in feature_dictionary.keys():
                    feature_dictionary[named_importances[i][0]]+=1
                else:
                    feature_dictionary[named_importances[i][0]]=1
    sorted_features = sorted(feature_dictionary, key=feature_dictionary.get, reverse=True)
    print(sorted_features)

    opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=25, scoring='roc_auc', random_state=42) 
    data = getter(datasheet=1)
    X, y = data.getXy_selectfeatures(columns=sorted_features[:20])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    opt.fit(X_train, y_train)
    
    xgboost_step = opt.best_estimator_.steps[0]
    xgboost_model = xgboost_step[1]
  
    print(opt.best_estimator_)
    print("Best Score: {}".format(opt.best_score_))
    print("Test Data Score: {}".format(opt.score(X_test, y_test)))

    data2 = getter(datasheet=1, group="V06")
    X2, y2 = data2.getXy_selectfeatures(columns=data.get_X_columns())
    print("V06 Data Score: {}".format(opt.score(X2, y2)))
    
    importances = opt.best_estimator_.steps[0][1].feature_importances_
    named_importances = list(zip(data.get_X_columns(), importances))
    sorted_feature_importances = sorted(named_importances, key=lambda item: item[1], reverse=True)
    for feature, importance in sorted_feature_importances:
        print(f"{feature}: {importance}")
    
    with open("best_features_model", 'wb') as file:
        pickle.dump((opt.best_score_, opt, None, data.get_X_columns()), file)
    
    opt = best_models[0][1]
    data = getter(datasheet=1)
    data2 = getter(datasheet=1, group="V06")
    X, y = data.getXy_selectfeatures(columns=best_models[0][3])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X2, y2 = data2.getXy_selectfeatures(columns=best_models[0][3])
    print(opt.best_estimator_)
    print("Best Score: {}".format(opt.best_score_))
    print("Test Data Score: {}".format(opt.score(X_test, y_test)))
    print("V06 Data Score: {}".format(opt.score(X2, y2)))

    importances = opt.best_estimator_.steps[0][1].feature_importances_
    named_importances = list(zip(data.get_X_columns(), importances))
    sorted_feature_importances = sorted(named_importances, key=lambda item: item[1], reverse=True)
    for feature, importance in sorted_feature_importances:
        print(f"{feature}: {importance}")
