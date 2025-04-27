import time
import pickle
from CSFData import getter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

loop_params = False

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


if loop_params:
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

                        # Add the current model to the list, then sort by scoring criteria and remove any that aren't a part of the top 20
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
                    
                        # Some time estimation metrics 
                        current_run+=1
                        elapsed = time.time() - loop_start
                        total_elapsed = time.time() - start_time
                        print(f"Iteration {current_run}: {elapsed:.2f} sec (Total elapsed: {total_elapsed:.2f} sec)")
                        print("Run {} of {}".format(current_run, num_runs))
                        print(f"Estimated time remaining: {((total_elapsed/current_run)*(num_runs-current_run)/60):.2f} min")

                        # Write to the pickle file
                        with open(pickle_name, 'wb') as file:
                            pickle.dump(best_models, file)


else:

    # Load the list of top 20 models
    pickle_name = "RF_best_models"
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
    opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=30, scoring='accuracy', random_state=42) 
    data = getter(datasheet=1)
    
    # Get the features from the data. Create a train test data set.
    # X, y = data.getXy_selectfeatures(columns=sorted_features[:20])
    X, y = data.getXy_selectfeatures(columns=sorted_features)
    # X, y = data.getXy_selectfeatures(columns=columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Fit the model
    opt.fit(X_train, y_train)
    
    RF_step = opt.best_estimator_.steps[0]
    RF_model = RF_step[1]
  
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
    with open("RF_best_features", 'wb') as file:
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


                            
best_model = best_models[0][1]
print(best_model.best_estimator_)
print(best_model.best_score_)
print(best_models[0][2])
