from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from CSFData import getter
import pickle
from Pipeline import PD_Pipeline


class eXtremeGradientBoost():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.estimators = [
            # ('imputer', KNNImputer()),
            # #('varthresh', VarianceThreshold()),
            # ('kselect', SelectKBest(mutual_info_classif)),
            ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective parameter
        ]

        self.search_space = {
            # 'clf__booster': ("dart", "gblinear"),
            # 'imputer__weights': Categorical({'uniform', 'distance'}),
            # 'imputer__n_neighbors': Integer(2, 20),
            #'varthresh__threshold': Real(0.0, 1.0),
            # 'kselect__k': Integer(5,20),
            # 'clf__min_child_weight': Real(0.0, 10),
            # 'clf__max_delta_step': Integer(0, 10),
            # 'clf__subsample': Real(0, 1),
            'clf__max_depth': Integer(2,8),
            # 'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
            # 'clf__subsample': Real(0.5, 1.0),
            # 'clf__colsample_bytree': Real(0.5, 1.0),
            # 'clf__colsample_bylevel': Real(0.5, 1.0),
            # 'clf__colsample_bynode' : Real(0.5, 1.0),
            'clf__reg_alpha': Real(0.0, 10.0),
            'clf__reg_lambda': Real(0.0, 10.0),
            'clf__gamma': Real(0.0, 10.0),
            # 'clf__tree_method': ("exact", "approx", "hist"),
            # 'clf__max_leaves': Integer(0, 10),
            # 'clf__num_parallel_tree': Integer(1,5)

        }

        self.selection_params = {
            'thresh': [0.2, 0.35, 0.5],
            'knn': [5,10,15],
            'varthresh': [0.5,0.8,1],
            'kselect': [20,25,30,40,50]
        }

    def run(
        self,
        name: str
    ):
        pipe = PD_Pipeline(
            name, 
            self.estimators, 
            self.search_space, 
            self.selection_params,
            **self.kwargs)
        pipe.run()