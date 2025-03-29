import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import graphviz
from matplotlib.patches import Patch
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, roc_auc_score

#NOTE: we wil look into removing rows that have a lot of imputed data values

class Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def featureSelector(self, threshold = 0.0, k = 15):
        """featureSelector returns a subset of k features(columns) from X, based on the
        value of threshold (used for sklearn.feature_selection VarianceThreshold) and
        mutual info classification.

        Parameters:
            X: Pandas DataFrame containing at least k fatures
            y: A list of the sample class attribures
            threshold: float; default 0. featues with variance less than this value will be ignored
            k: int; number of features to be selected
        
        """
        sel = VarianceThreshold(threshold=threshold)
        sel.set_output(transform="pandas")
        self.X = sel.fit_transform(self.X, self.y) #Removes low variance features

        sel = SelectKBest(mutual_info_classif, k=k)
        mic_Parameters = sel.get_Parameters()
        mic_Parameters["random_state"] = 42
        sel.set_output(transform="pandas")
        self.X = sel.fit_transform(X, y) #Removes features based on mutual info classifer
    
    def rf(self, n=1000, n_splits=5):
        """rf (Random Forest) prints the training and testing scores of a 'n' tree random 
        forest classification model, using cross validation with Stratified K Folding. Scores are based on 
        roc_auc metric

        Parameters:
            X: Pandas DataFrame containing feature data
            y: A list of the sample class attribures
            n: Default 1000; number of random trees to be generated
            n_splits: Default 5; number of stratified folds to use for cross validation.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=y)

        clf = RandomForestClassifier(random_state=42, n_estimators=n)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = cross_validate(clf, X_train, y_train, cv=cv, scoring='roc_auc', return_train_score=True)
        print(cv_scores["train_score"])
        print(cv_scores["test_score"])
        
    def svc(self, k='rbf', deg=2, C=1000, gamma=.001, n_splits=5):
        """svc (Support Vector Classification) prints the training and testing scores of a Support Vector
        classification model, using cross validation with Stratified K Folding. Scores are based on 
        roc_auc metric

        Parameters:
            X: Pandas DataFrame containing feature data
            y: A list of the sample class attribures
            deg: Default 2; degree for polynomial kernal
            C: Default 1000; C value for SVC
            gamma: Default .001; gamma value for rbf kernal
            n_splits: Default 5; number of stratified folds to use for cross validation.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=y)

        clf = svm.SVC(C=C, gamma=gamma, kernel=k, degree=deg, probability=True)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_validate(clf, X_train, y_train, cv=cv, scoring='roc_auc', return_train_score=True)
        print(cv_scores["train_score"])
        print(cv_scores["test_score"])

    def xgboost(self):
        mapping = {'Control': 0, 'PD': 1}
        xgb_y = [mapping.get(item, item) for item in self.y]
        X_train, X_test, y_train, y_test = train_test_split(self.X, xgb_y, test_size=0.2, random_state=42, stratify=y)

        clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        ax = xgb.plot_tree(clf, num_trees=2)
        fig = ax.figure
        fig.add_axes(ax)
        plt.show()


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





def getdata(path="CSF-Research-Project\Data\FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", datasheet=3):
    """ 
    getdata reads from the CSF excel data sheet and returns a Dataframe with the Class data attribute
    attached.

    Parameters:
        path: Default is the Data folder with the data in it
        datasheet: int 1-3; refers to the sheet within the excel document that the data is to be pulled from
            1: Batch-normalized Data
            2: Batch-norm Imputed Data
            3: Log Transformed Data
    Returns:
        DataFrame containing with column 0 as the class data and all following columns as attributes
    """
    mapping = {1:'Batch-normalized Data', 2: 'Batch-norm Imputed Data', 3:'Log Transformed Data'}
    patient_data = pd.read_excel(
        path,
        sheet_name = "Sample Meta Data",
        header=0,
        usecols = ["PARENT_SAMPLE_NAME", "COHORT", "PPMI_CLINICAL_EVENT", "PPMI_COHORT"],
        index_col="PARENT_SAMPLE_NAME"
    )
    patient_data = patient_data.drop(patient_data[
        (patient_data.COHORT != "PPMI") |
        (patient_data.PPMI_CLINICAL_EVENT != "BL")
    ].index)
    patient_data = patient_data.drop("COHORT", axis=1)
    patient_data = patient_data.drop("PPMI_CLINICAL_EVENT", axis=1)

    df = pd.read_excel(
        path,
        sheet_name = mapping.get(datasheet),
        index_col="PARENT_SAMPLE_NAME"
    )

    df = patient_data.join(df, on="PARENT_SAMPLE_NAME", how='inner')
    return df


data = getdata(datasheet=2)




# data = pd.read_csv("CSF-Research-Project\Data\PPMI_Cohort_Filtered.csv")
# data = data.drop(data.columns[0], axis=1)   #Drop the first column, which is the sequential numbers from R


y = data["PPMI_COHORT"]
X = data.drop(data.columns[0:2], axis=1)

m = Model(X, y)
m.featureSelector(.2)
m.rf()

