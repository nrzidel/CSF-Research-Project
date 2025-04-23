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

#TODO get the True/False positives for a whole dataset (confusion matrix)

#NOTE: we wil look into removing rows that have a lot of imputed data values

pickle_name = "RF_best_models_roc_auc"
with open(pickle_name, 'rb') as file:
    best_models = pickle.load(file)

for model in best_models:
    print(model[1].best_score_)
    print(model[1].best_estimator_)

estimators = [
    ('imputer', KNNImputer()),
    ('norm', Normalizer()),
    ('kselect', SelectKBest(mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
]
pipe = Pipeline(steps=estimators)

search_space = {
    'imputer__weights': Categorical({'uniform', 'distance'}),
    'imputer__n_neighbors': Integer(2, 20),
    'norm__norm': Categorical({'l1', 'l2', 'max'}),
    'kselect__k': Integer(10,20),
    'rf__n_estimators': Integer(50, 500),
    'rf__criterion': Categorical({'gini', 'entropy', 'log_loss'}),
    'rf__ccp_alpha': Real(0.0, 0.25)
    }


scorers = ['roc_auc', 'accuracy']

for scorer in scorers:
    best_models = []
    print(f"Entering Loop for {scorer}")

    pickle_name = f"RF_best_models_{scorer}"
    for thresh in [0.2, 0.3, 0.4, 0.5]:
        opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=30, scoring=scorer, random_state=42, n_jobs=2) 

        data = getter(datasheet=1)
        X, y = data.getXy(nathresh=thresh)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        opt.fit(X_train, y_train)
        print(y_test)
        print(opt.predict(X_test))
        print(opt.score(X_test, y_test))
        features = opt.best_estimator_[:-1].get_feature_names_out()

        best_models.append([
            opt.best_score_, 
            opt, 
            thresh, 
            data.get_X_columns()
            ])
        best_models = sorted(best_models, key=lambda x: x[0], reverse=True)
        if len(best_models)>10:
            best_models = best_models[:10]


        #Leave one Out Testing
        model = opt.best_estimator_
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring=scorer)
        print("LOOCV mean score:", scores.mean())
    
        with open(pickle_name, 'wb') as file:
            pickle.dump(best_models, file)




                            
best_model = best_models[0][1]
print(best_model.best_estimator_)
print(best_model.best_score_)
print(best_models[0][2])
