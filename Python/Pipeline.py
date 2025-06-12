import time
import pickle
from CSFData import getter
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


class PD_Pipeline():

    def __init__(
            self,
            title: str,
            estimators: list,
            search_space: dict,
            selection_params: dict     
        ):
        
        self.title = title
        self.pipe = Pipeline(steps=estimators)
        self.search_space = search_space
        self.selection_params = selection_params

        example_selection_params = {
            'thresh': [],
            'knn': [],
            'varthresh': [],
            'kselect': []
        }

    
    def run(self, **kwargs):
        num_runs = 1
        current_run = 0
        best_models = []
        start_time = time.time()

        # Count the number of combinations of selections
        for items in self.selection_params.values():
            num_runs *= len(items)

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


            opt = BayesSearchCV(self.pipe, self.search_space, cv=10, n_iter=3, scoring='accuracy', random_state=42, n_jobs=2) 

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
            print(f"Iteration {current_run}: {elapsed:.2f} sec (Total elapsed: {total_elapsed:.2f} sec)")
            print("Run {} of {}".format(current_run, num_runs))
            print(f"Estimated time remaining: {((total_elapsed/current_run)*(num_runs-current_run)/60):.2f} min")

            # Write to the pickle file
            with open(self.title, 'wb') as file:
                pickle.dump(best_models, file)


                        



