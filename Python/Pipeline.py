import time
import pickle
from CSFData import getter
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


class PD_Pipeline():

    def __init__(
            self,
            estimators: list,
            search_space: dict,
            selection_params: dict, 
            title: str = "default_model_name",
            **kwargs     
        ):
        self.kwargs = kwargs
        self.title = title
        self.pipe = Pipeline(steps=estimators)
        self.search_space = search_space
        self.selection_params = selection_params
        self.test_size = kwargs['config']['model_settings'].getfloat('test_size')

        example_selection_params = {
            'thresh': [],
            'knn': [],
            'varthresh': [],
            'kselect': []
        }


    def run(self):
        num_runs = 1
        current_run = 0
        best_models = []
        start_time = time.time()

        # Count the number of combinations of selections
        for items in self.selection_params.values():
            num_runs *= len(items)

        print(num_runs)

        selection_keys = self.selection_params.keys()
        selection_lists = self.selection_params.values()

        for combo in product(*selection_lists):

            loop_start = time.time()
            keyed_combo = dict(zip(selection_keys, combo))

            # TODO: this should be updated in the future to work with different feature selection methods
            # For now I have hard coded it to work with the existng functions

            thresh = keyed_combo['thresh']
            knn = keyed_combo['knn']
            varthresh = keyed_combo['varthresh']
            kselect = keyed_combo['kselect']

            # --------------------------------------------------


            opt = BayesSearchCV(self.pipe, self.search_space, cv=10, n_iter=25, scoring='accuracy', random_state=42, n_jobs=2) 

            # BL Data
            data = getter(datasheet=1, 
                          group='BL',
                          **self.kwargs
                          )
            X, y = data.getXy(nathresh=thresh, knn = knn, varthresh=varthresh, kselect=kselect)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, stratify=y)
        
            # V06 Data
            data2 = getter(datasheet=1, 
                          group='V06',
                          **self.kwargs
                          )
            X2, y2 = data2.getXy_selectfeatures(columns=data.get_X_columns())

            # Fit then score against test and V06
            opt.fit(X_train, y_train)
            test_score = opt.score(X_test, y_test)
            V06_score = opt.score(X2, y2)

            print(y_test)
            print(opt.predict(X_test))
            print(test_score)

            # Add the current model to the list, then sort by scoring criteria and remove any that aren't a part of the top 20
            best_models.append([
                opt.best_score_, 
                opt, 
                (1, thresh, knn, varthresh, kselect), 
                data.get_X_columns(),
                test_score,
                V06_score
                ])
            best_models = sorted(best_models, key=lambda x: (x[5],x[4],x[0]), reverse=True)
            if len(best_models)>20:
                best_models = best_models[:20]
        
            # Some time estimation metrics 
            current_run+=1
            elapsed = time.time() - loop_start
            total_elapsed = time.time() - start_time

            self.kwargs['logger'].info(f"Iteration {current_run}: {elapsed:.2f} sec (Total elapsed: {total_elapsed:.2f} sec) \n"
                                  f"Run {current_run} of {num_runs}\n"
                                  f"Estimated time remaining: {((total_elapsed/current_run)*(num_runs-current_run)/60):.2f} min\n"
            )
        # Write to the pickle file
        with open(f'Python/picklejar/{self.title}.pickle', 'wb') as file:
            pickle.dump(best_models, file)

    def frequent_features(self, name: str = None):
        # Load the list of top 20 models
        with open(f'Python/picklejar/{name}.pickle', 'rb') as file:
            best_models = pickle.load(file)

        # Create a dictonary of all of the features in each model's feature importances.
        # Record how often a feature is used.
        feature_dictionary = {}
        for model in best_models:
            model_obj = model[1]
            x_columns = model[3]
            importances = model_obj.best_estimator_.steps[0][1].feature_importances_
            named_importances = list(zip(x_columns, importances))
            for i in range(len(named_importances)):
                feature = named_importances[i][0]
                importance = named_importances[i][1]
                if feature in feature_dictionary:
                    feature_dictionary[feature]['count'] += 1
                    feature_dictionary[feature]['total_importance'] += importance
                else:
                    feature_dictionary[feature] = {
                        'count': 1,
                        'total_importance': importance
                    }

        # Calculate average importance for each feature
        for feature, stats in feature_dictionary.items():
            stats['average_importance'] = stats['total_importance'] / stats['count']

        # Sort features by count and then by average importance
        sorted_features = sorted(feature_dictionary.items(), key=lambda x: (x[1]['count'], x[1]['average_importance']), reverse=True)
            # Example structure of sorted_features:
            # [
            #   ('feature1', {'count': 5, 'total_importance': 2.34, 'average_importance': 0.468}),
            #   ('feature2', {'count': 3, 'total_importance': 1.20, 'average_importance': 0.400}),
            #   ...
            # ]



        # Limit to maximum number of features if specified
        max_features = self.kwargs['config']['model_settings'].getint('maximum_frequent_features')
        if max_features is not None:
            sorted_features = sorted_features[:max_features]

        opt = BayesSearchCV(self.pipe, self.search_space, cv=10, n_iter=50, scoring='roc_auc', random_state=42)
        data = getter(datasheet=1, 
                      group='BL', 
                      **self.kwargs)

        # Get the features from the data. Create a train test data set.
        X, y = data.getXy_selectfeatures(columns=[feature[0] for feature in sorted_features])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, stratify=y, random_state=42)

        # Fit the model on BL data
        opt.fit(X_train, y_train)
        BL_test_score = opt.score(X_test, y_test)

        self.kwargs['logger'].info(opt.best_estimator_)
        self.kwargs['logger'].info(f"Best Score: {opt.best_score_}")
        self.kwargs['logger'].info(f"Test Data Score: {BL_test_score}")

        # V06 Data
        data_V06 = getter(datasheet=1, 
                       group="V06", 
                       **self.kwargs)
        X_v06, y_v06 = data_V06.getXy_selectfeatures(columns=data.get_X_columns())
        V06_test_score = opt.score(X_v06, y_v06)
        self.kwargs['logger'].info(f"V06 Data Score: {V06_test_score}")

        frequent_features_model = [
            opt.best_score_, 
            opt,
            None,
            data.get_X_columns(),
            BL_test_score,
            V06_test_score
        ]

        # Store the frequent features model.
        with open(f"Python/picklejar/frequent pickles/{name} Frequent Features.pickle", 'wb') as file:
            pickle.dump([frequent_features_model], file)



