import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import graphviz
from CSFData import getter
from matplotlib.patches import Patch
from sklearn import clone, svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn.impute import KNNImputer


#NOTE: we wil look into removing rows that have a lot of imputed data values

class Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def rf(self, n=1000, n_splits=5):
        """rf (Random Forest) prints the training and testing scores of a 'n' tree random 
        forest classification model, using cross validation with Stratified K Folding. Scores are based on 
        roc_auc metric

        Parameters:
            n: Default 1000; number of random trees to be generated
            n_splits: Default 5; number of stratified folds to use for cross validation.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=y)

        clf = RandomForestClassifier(random_state=42, n_estimators=n)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_results = cross_validate(clf, X_train, y_train, cv=cv, scoring='roc_auc', return_train_score=True, return_estimator=True)
        cv_clf = cv_results['estimator']

        print(self.testModel(X_test, y_test, cv_clf, n_splits))

        #print(cv_results['estimator'])
        #print(cv_results["train_score"])
        #print(cv_results["test_score"])
        
    def svc(self, k='rbf', deg=2, C=1000, gamma=.001, n_splits=5):
        """svc (Support Vector Classification) prints the training and testing scores of a Support Vector
        classification model, using cross validation with Stratified K Folding. Scores are based on 
        roc_auc metric

        Parameters:
            deg: Default 2; degree for polynomial kernal
            C: Default 1000; C value for SVC
            gamma: Default .001; gamma value for rbf kernal
            n_splits: Default 5; number of stratified folds to use for cross validation.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=y)

        clf = svm.SVC(C=C, gamma=gamma, kernel=k, degree=deg, probability=True)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(clf, X_train, y_train, cv=cv, scoring='roc_auc', return_train_score=True, return_estimator=True)
        cv_clf = cv_results['estimator']

        print(self.testModel(X_test, y_test, cv_clf, n_splits))
        
        # print(cv_results["train_score"])
        # print(cv_results["test_score"])

    def xgboost(self, n_splits=5):
        # TODO: xgboost has numerous parameters which can (and probably should) be modified during optimization
        """
        xgboost prints the training and testing scores of an xgboost model using Stratified K Fold
        cross validation

        Parameters: 
            n_splits: int Default=5; number of stratified folds to use for cross validation.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=y)

        clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=5, reg_alpha=2.0)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_results = self.xgbCrossValidate(clf, X_train, y_train, cv)
        cv_clf = cv_results['estimator']

        print(self.testModel(X_test, y_test, cv_clf, n_splits))

    def xgbCrossValidate(self, estimator, X_in, y_in, cv):
        """
        xgbCrossValidate is a custom cross validation method used exclusively for xgboost. Per the xgboost
        documentaion, 'using early stopping during cross validation may not be a perfect approach because 
        it changes the model’s number of trees for each validation fold, leading to different model.' The
        following is a modified version of the sample given on the documentation page. Visit -> 
        https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html <- to see the example.

        Parameters:
            estimator: XGBClassifier; the classifier object to be fitted and tested. Note: this estimator
                will be cloned before being passed to fit_and_score().
            X_in; training data set (X_train), naming convention was modified to avoid confusion within the method.
            y_in; class data set (y_train), naming convention was modified to avoid confusion within the method.
            cv; Cross Validator Object.
        """
        cv_results = {'estimator':[], 'train_score':[], 'test_score':[]}

        for train, test in cv.split(X_in, y_in):
            X_train = X_in.iloc[train]
            X_test = X_in.iloc[test]
            y_train = y_in[train]
            y_test = y_in[test]
            est, train_score, test_score = self.fit_and_score(
                clone(estimator), X_train, X_test, y_train, y_test
            )
            cv_results['estimator'].append(est)
            cv_results['train_score'].append(train_score)
            cv_results['test_score'].append(test_score)

        return cv_results

    def fit_and_score(self, estimator, X_train, X_test, y_train, y_test):
        """
        fit_and_score fits the XGBClassifier and returns the fitted estimator, training and testing scores

        Parameters:
            estimator: UNFITTED XGBClassifier
            X_Train: training feature data set.
            X_Test: testing feature data set.
            y_train: training class data set.
            y_test: testing class data set.
        """
        estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        train_score = estimator.score(X_train, y_train)
        test_score = estimator.score(X_test, y_test)

        return estimator, train_score, test_score

    def testModel(self, X_test, y_test, cv_clf, n_splits):
        """
        testModel tests the classification model against the testing data. 
        """
        average_accuracy = 0
        for c in cv_clf:
            y_predicted = c.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_predicted).ravel()
            average_accuracy += (tp+tn)/(tp+tn+fp+fn)
            print(f'TN: {tn} FP: {fp} FN: {fn} TP: {tp} Training accuracy: {(tp+tn)/(tp+tn+fp+fn):.2f}')
        return(average_accuracy/n_splits)


class dataVisualization:
    def varianceThresholdGraph(X, y):
        thresholds = [None] * 20            # List of variance thresholds tested
        num_features_removed = [None] * 20  # List of number of features removed for each threshold

        for i in range(20):
            sel = VarianceThreshold(threshold=(i*.05))
            sel.set_output(transform="pandas")
            X_features_before = X.shape[1]
            print(X_features_before)
            X_new = sel.fit_transform(X, y)     # Removes i variance features
            X_features_after = X_new.shape[1]
            thresholds[i] = (i*.05)
            num_features_removed[i] = (X_features_before - X_features_after)

        print(thresholds)
        print(num_features_removed)

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, num_features_removed, marker='o', linestyle='-')
        plt.xlabel('Variance Threshold')
        plt.ylabel('Number of Features Removed')
        plt.title('Effect of Variance Threshold on Feature Removal')
        plt.grid(True)
        plt.show(block = True)  

        #varianceThresholdGraph(X, y)

    def boxNWhisker(X, y, i=0):
        class_1_features = X.iloc[(y == 'PD'), i]  # List of feature values for Class 1
        class_2_features = X.iloc[(y == 'Control'), i]  # List of feature values for Class 2
        colors = ['lightblue', 'lightgreen'] 

        plt.figure(figsize=(8, 5))
        box = plt.boxplot([class_1_features, class_2_features], labels=['Parkinson\'s', 'Control'], patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        legend_patches = [Patch(color=colors[0], label='Parkinson\'s'), Patch(color=colors[1], label='Control')]
        plt.legend(handles=legend_patches, loc='upper right') 
        plt.xlabel('Class')
        plt.ylabel('Log Transformed Value')
        plt.title(X.columns[i])
        plt.grid(True)
        plt.show()

estimators = [
    ('imputer', KNNImputer()),
    ('varthresh', VarianceThreshold()),
    ('kselect', SelectKBest(mutual_info_classif)),
    ('rf', RandomForestClassifier(random_state=42))
    #('svc', svm.SVC(random_state=42))
]
pipe = Pipeline(steps=estimators)

search_space = {
    'imputer__weights': Categorical({'uniform', 'distance'}),
    'imputer__n_neighbors': Integer(2, 20),
    'varthresh__threshold': Real(0.0, 1.0),
    'kselect__k': Integer(5,20),
    'rf__n_estimators': Integer(500, 2000),
    'rf__criterion': Categorical({'gini', 'entropy', 'log_loss'}),
    'rf__max_depth': Integer(2, 20),
    'rf__min_samples_split': Integer(2, 10),
    'rf__min_samples_leaf': Integer(1, 10),
    'rf__min_weight_fraction_leaf': Real(0.0, 0.5),
    'rf__min_impurity_decrease': Real(0.0, 10.0),
    'rf__ccp_alpha': Real(0.0, 1.0)}

pickle_name = "RF_sheet_1"

print("Entering Loop:")

best_models = []

for thresh in [0.5]:
    opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=150, scoring='roc_auc', random_state=42, n_jobs=3) 

    data = getter(datasheet=1)
    X, y = data.getXy(nathresh=thresh)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    opt.fit(X_train, y_train)

    y_score = opt.score(X_test, y_test)
    print(f"threshold: {thresh} \n score: {y_score}")

    best_models.append([y_score, opt, thresh, data.get_X_columns()])
    best_models = sorted(best_models, key=lambda x: x[0], reverse=True)
    if len(best_models)>10:
        best_models = best_models[:10]
    
with open(pickle_name, 'wb') as file:
    pickle.dump(best_models, file)
                            
best_model = best_models[0][1]
print(best_model.best_estimator_)
print(best_model.best_score_)
print(best_model.score(X_test, y_test))
print(best_models[0][2])
