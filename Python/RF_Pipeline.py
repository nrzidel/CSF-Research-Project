from Pipeline import PD_Pipeline
from CSFData import getter
from sklearn.ensemble import RandomForestClassifier 
from skopt.space import Real, Categorical, Integer


estimators = [
    # ('imputer', KNNImputer()),
    # ('norm', Normalizer()),
    # ('kselect', SelectKBest(mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
]

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

selection_params = {
    'thresh': [0.2, 0.35, 0.5],
    'knn': [5,10,15],
    'varthresh': [0.5,0.8,1],
    'kselect': [20,25,30,40,50]
}

pipe = PD_Pipeline('tester', estimators, search_space, selection_params)
pipe.run()
