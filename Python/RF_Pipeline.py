from sklearn.ensemble import RandomForestClassifier 
from skopt.space import Real, Categorical, Integer

from Pipeline import PD_Pipeline

class RandomForest(PD_Pipeline):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.estimators = [
            # ('imputer', KNNImputer()),
            # ('norm', Normalizer()),
            # ('kselect', SelectKBest(mutual_info_classif)),
            ('rf', RandomForestClassifier(random_state=42))
        ]

        self.search_space = {
            # 'imputer__weights': Categorical({'uniform', 'distance'}),
            # 'imputer__n_neighbors': Integer(2, 20),
            # 'norm__norm': Categorical({'l1', 'l2', 'max'}),
            # 'kselect__k': Integer(10,20),
            'rf__n_estimators': Integer(50, 500),
            'rf__criterion': Categorical({'gini', 'entropy', 'log_loss'}),
            'rf__ccp_alpha': Real(0.0, 0.25),
            'rf__max_depth': Integer(5, 15)
            }

        self.selection_params = {
            'thresh': [0.2, 0.35, 0.5],
            'knn': [5,10,15],
            'varthresh': [0.5,0.8,1],
            'kselect': [20,25,30,40,50]
        }
        # Call the parent constructor if needed
        super().__init__(
            estimators=self.estimators,
            search_space=self.search_space,
            selection_params=self.selection_params,
            **kwargs
        )
    
    def run(self, name: str):
        # Call the parent class's run method
        super().run(name)

    def frequent_features(self, name: str = None):
        if name is None:
            name = input('Run frequent features on which pickle? ')

        # Call the parent class's frequent_features method
        super().frequent_features(name)
