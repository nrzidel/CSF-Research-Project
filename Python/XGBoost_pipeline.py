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


class Process_Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def cleanData(self, nathresh=.5, k=5):
            """
            cleanData modifies Model.X to remove columns containing more than nathresh % missing values. 
            Any missing values that are not removed are imputed using K Nearest Neigbors.

            Parameters:
                nathresh: Double Default = .5 (50%); threshold of missing values permitted before a column is dropped
                k: int Default = 5; number of neigbors to be used for KNNImputer
            """

            nathresh = .5        # % of samples allowed to be NA before column is dropped
            self.X = self.X.dropna(axis=1, thresh=int((1 - nathresh) * self.X.shape[0]))

            #NOTE: Optimize k value

            imputer = KNNImputer(weights='distance', n_neighbors=k)
            imputer.set_output(transform='pandas')
            self.X = imputer.fit_transform(self.X, self.y)

    def featureSelector(self, threshold = 0.0, k = 20):
        """featureSelector returns a subset of k features(columns) from X, based on the
        value of threshold (used for sklearn.feature_selection VarianceThreshold) and
        mutual info classification.

        Parameters:
            threshold: float; default 0. featues with variance less than this value will be ignored
            k: int; number of features to be selected
        
        """
        sel = VarianceThreshold(threshold=threshold)
        sel.set_output(transform="pandas")
        self.X = sel.fit_transform(self.X, self.y) #Removes low variance features

        sel = SelectKBest(mutual_info_classif, k=k)
        mic_params = sel.get_params()
        mic_params["random_state"] = 42
        sel.set_output(transform="pandas")
        self.X = sel.fit_transform(self.X, self.y) #Removes features based on mutual info classifer
    
    def get_data(self):
        return self.X, self.y 
    

data = pd.read_csv("Data\PPMI_Cohort_Filtered.csv")
data = data.drop(data.columns[0], axis=1)   #Drop the first column, which is the sequential numbers from R

y = data["PPMI_COHORT"].values
X = data.iloc[:, 3:]

le = LabelEncoder()
y = le.fit_transform(y)

data = Process_Data(X,y)
data.cleanData()
data.featureSelector(0.9,50)
X, y = data.get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

estimators = [
    # ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
]
pipe = Pipeline(steps=estimators)

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=15, scoring='roc_auc', random_state=42) 
# in reality, you may consider setting cv and n_iter to higher values
opt.fit(X_train, y_train)
print(opt.best_estimator_)
print(opt.best_score_)
print(opt.score(X_test, y_test))
# opt.predict(X_test)
# opt.predict_proba(X_test)
# opt.best_estimator_.steps
xgboost_step = opt.best_estimator_.steps[0]
xgboost_model = xgboost_step[1]
plot_importance(xgboost_model)

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
