import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import graphviz
from CSFData import getter
from matplotlib.patches import Patch
import pickle
from sklearn import clone, svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance

estimators = [
    # ('imputer', KNNImputer()),
    # ('norm', Normalizer()),
    # ('kselect', SelectKBest(mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
]
pipe = Pipeline(steps=estimators)

search_space = {
    # 'imputer__weights': Categorical({'uniform', 'distance'}),
    # 'imputer__n_neighbors': Integer(2, 20),
    # 'norm__norm': Categorical({'l1', 'l2', 'max'}),
    # 'kselect__k': Integer(10,20),
    'rf__n_estimators': Integer(50, 500),
    'rf__criterion': Categorical({'gini', 'entropy', 'log_loss'}),
    'rf__ccp_alpha': Real(0.0, 0.25),
    'rf__max_depth': Integer(5, 15)
    }

pickle_name = "RF_best_models"
best_models = []
num_runs = 135
current_run = 0
start_time = time.time()

print("Entering Loop:")
for sheet in [1]:
    for thresh in [0.2, 0.35, 0.5]:
        for knn in [5, 10, 15]:
            for varthresh in [.5, .8, 1]:
                for kselect in [20, 25, 30, 40, 50]:
                    loop_start = time.time()
                    opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=30, scoring='accuracy', random_state=42, n_jobs=2) 

                    # BL Data
                    data = getter(datasheet=1)
                    X, y = data.getXy(nathresh=thresh, knn = knn, varthresh=varthresh, kselect=kselect)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # V06 Data
                    data2 = getter(datasheet=1, group="V06")
                    X2, y2 = data2.getXy_selectfeatures(columns=data.get_X_columns())

                    opt.fit(X_train, y_train)
                    test_score = opt.score(X_test, y_test)
                    V06_score = opt.score(X2, y2)

                    print(y_test)
                    print(opt.predict(X_test))
                    print(test_score)

                    best_models.append([
                        opt.best_score_, 
                        opt, 
                        (sheet, thresh, knn, varthresh, kselect), 
                        data.get_X_columns(),
                        test_score,
                        V06_score
                        ])
                    best_models = sorted(best_models, key=lambda x: (x[5],x[4],x[0]), reverse=True)
                    if len(best_models)>20:
                        best_models = best_models[:20]
                    
                    current_run+=1
                    elapsed = time.time() - loop_start
                    total_elapsed = time.time() - start_time
                    print(f"Iteration {current_run}: {elapsed:.2f} sec (Total elapsed: {total_elapsed:.2f} sec)")
                    print("Run {} of {}".format(current_run, num_runs))
                    print(f"Estimated time remaining: {((total_elapsed/current_run)*(num_runs-current_run)/60):.2f} min")

                    with open(pickle_name, 'wb') as file:
                        pickle.dump(best_models, file)




                            
best_model = best_models[0][1]
print(best_model.best_estimator_)
print(best_model.best_score_)
print(best_models[0][2])
