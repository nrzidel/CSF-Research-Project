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
    # ('imputer', KNNImputer()),
    # #('varthresh', VarianceThreshold()),
    # ('kselect', SelectKBest(mutual_info_classif)),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)

search_space = {
    # 'clf__booster': ("dart", "gblinear"),
    # 'imputer__weights': Categorical({'uniform', 'distance'}),
    # 'imputer__n_neighbors': Integer(2, 20),
    #'varthresh__threshold': Real(0.0, 1.0),
    # 'kselect__k': Integer(5,20),
    'clf__min_child_weight': Real(0.0, 10),
    'clf__max_delta_step': Integer(0, 10),
    'clf__subsample': Real(0, 1),
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    # 'clf__subsample': Real(0.5, 1.0),
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

if loop_params: # Boolean to cgenearate model list or to post process the list

    pickle_name = "xgboost_sheet_1_n_iter_25_roc_auc_combined_score_cv10_v7"
    num_runs = 135
    current_run = 0
    # Parameter sweeps are selected based on results from previous runs. This set of 135 is reasonable for running experiments: 1-2 hours.
    for sheet in [1]:
        for thresh in [0.2, 0.35, 0.5]:
            for knn in [5,10,15]:
                for varthresh in [0.5,0.8,1]:
                    for kselect in [20,25,30,40,50]:

                        # Use Bayes Search to optimize the XGBoost model with the above defined parameter sweeps.
                        opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=25, scoring='accuracy', random_state=42) 
                        # get data object 
                        data = getter(datasheet=sheet)
                        # select features by the sweep inputs
                        X, y = data.getXy(nathresh=thresh, knn = knn, varthresh=varthresh, kselect=kselect)
                        # Get training test, train set
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)       
                        # fit the model to the training data
                        opt.fit(X_train, y_train)
                        # Get the V06 test data set
                        data2 = getter(datasheet=1, group="V06")
                        # Select the same columns as the train data set
                        X2, y2 = data2.getXy_selectfeatures(columns=data.get_X_columns())
                        # Score the model
                        test_score = opt.score(X_test, y_test)
                        v06_score = opt.score(X2, y2)

                        # Append results to a tuple for later analysis
                        best_models.append((opt.best_score_, opt, (sheet, thresh, knn, varthresh, kselect), data.get_X_columns(), test_score, v06_score))
                        # Sort the models by results on V06 test data, BL test data, and best_score (training data cross validation) in that order. This prioirtizes results on test data to prevent selection of overfitted models. 
                        best_models = sorted(best_models, key=lambda x: (x[5],x[4],x[0]), reverse=True)
                        
                        # keep only the top 20 models. No need to store them all.
                        if len(best_models)>20:
                            best_models = best_models[:20]
                        
                        current_run+=1
                        print("Run {} of {}".format(current_run, num_runs))
        
    # Store the list of tuples to pickle file for later
    with open(pickle_name, 'wb') as file:
        pickle.dump(best_models, file)
                            
    # Print the result of the best model.
    best_model = best_models[0][1]
    print(best_model.best_estimator_)
    print(best_model.best_score_)
    # print(best_model.score(X_test, y_test))
    print(best_models[0][2])

else:

    # Load the list of top 20 models
    pickle_name = "xgboost_sheet_1_n_iter_25_accuracy_best_score_cv10_v7"
    with open(pickle_name, 'rb') as file:
        best_models = pickle.load(file)

    # Create a dictonary of all of the features in each model's feature importances.
    # Record how often a feature is used.
    feature_dictionary = {}
    for model in best_models:
        model_obj = model[1]
        x_columns = model[3]
        importances = model_obj.best_estimator_.steps[0][1].feature_importances_
        named_importances = list(zip(x_columns, importances))
        print("Best_Score: {}".format(model[0]))
        print(model_obj.best_estimator_)    
        print(named_importances)
        print(model[2])
        for i in range(len(named_importances)):
            if named_importances[i][0] in feature_dictionary.keys():
                feature_dictionary[named_importances[i][0]]+=1
            else:
                feature_dictionary[named_importances[i][0]]=1
    
    # Limit the feature set to the features that show up the most. A cutoff of 6 was selected to get a set of 39 frequent features.
    best_features = {k:v for k, v in feature_dictionary.items() if v>=6}

    # Sort the features and print
    sorted_features = sorted(best_features, key=best_features.get, reverse=True)
    print(sorted_features)

    # Create a bayes optimization opt for the new model
    opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=50, scoring='roc_auc', random_state=42) 
    data = getter(datasheet=1)
    
    # Get the features from the data. Create a train test data set.
    # X, y = data.getXy_selectfeatures(columns=sorted_features[:20])
    X, y = data.getXy_selectfeatures(columns=sorted_features)
    # X, y = data.getXy_selectfeatures(columns=columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Fit the model
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
    
    # Store the frequent features model.
    with open("best_features_model_frequency_test", 'wb') as file:
        pickle.dump([(opt.best_score_, opt, None, data.get_X_columns())], file)
    
    opt = best_models[0][1]
    data = getter(datasheet=1)
    data2 = getter(datasheet=1, group="V06")
    X, y = data.getXy_selectfeatures(columns=best_models[0][3])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X2, y2 = data2.getXy_selectfeatures(columns=best_models[0][3])
    
    # Print the results.
    print(opt.best_estimator_)
    print("Best Score: {}".format(opt.best_score_))
    print("Test Data Score: {}".format(opt.score(X_test, y_test)))
    print("V06 Data Score: {}".format(opt.score(X2, y2)))

    importances = opt.best_estimator_.steps[0][1].feature_importances_
    named_importances = list(zip(data.get_X_columns(), importances))
    sorted_feature_importances = sorted(named_importances, key=lambda item: item[1], reverse=True)
    for feature, importance in sorted_feature_importances:
        print(f"{feature}: {importance}")

# ['100001178', '100004634', '100001992', '100006191', '999923644', '100001403', '100004208', '100001108', '100001605', '100021467', '100000894', '849', '100002927', '501', '999911299']
