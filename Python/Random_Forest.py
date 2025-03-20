import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay


data = pd.read_csv("CSF-Research-Project\Data\PPMI_Cohort_Filtered.csv")
data = data.drop(data.columns[0], axis=1)   #Drop the first column, which is the sequential numbers from R

y = data["PPMI_COHORT"].values
X = data.drop(data.columns[0:2], axis=1)

sel = VarianceThreshold(.8)
sel.set_output(transform="pandas")
X = sel.fit_transform(X, y) #Removes low variance features

sel = SelectKBest(mutual_info_classif, k=17)
sel.set_output(transform="pandas")
X = sel.fit_transform(X, y) #Removes features based on mutual info classifer




#NOTE: Professor said we may need to still perform normalization on the data


d = preprocessing.normalize(X, axis=0)
#X = pd.DataFrame(d, columns=X.columns)





#X = SelectKBest(f_classif, k=50).fit_transform(X, y)

#print(X.head())
#print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def rf(X_train, X_test, y_train, y_test, n=1000):
    clf = RandomForestClassifier(random_state=42, n_estimators=n)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

    clf = clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]


    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label='PD')
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Random Forest")
    plt.legend()
    plt.show()

    #print(clf.predict_proba(X_test))
    #print(y_scores)
    print("Cross-validation scores Random Forest:", cv_scores)
    print("Mean accuracy:", np.mean(cv_scores))

    y_pred = clf.predict(X_train) #Training data confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    print(f'Training data -> TN: {tn} FP: {fp} FN: {fn} TP: {tp} Training accuracy: {(tp+tn)/(tp+tn+fp+fn):.2f}')

    y_pred = clf.predict(X_test) #Testing data confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Testing data -> TN: {tn} FP: {fp} FN: {fn} TP: {tp} Testing accuracy: {(tp+tn)/(tp+tn+fp+fn):.2f}')

def svc(X_train, X_test, y_train, y_test, k, deg=2):
    C=1e+03 #1000
    gamma=1e-02 #0.0001
    clf = svm.SVC(C=C, gamma=gamma, kernel=k, degree=deg, probability=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

    clf = clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]

    print("Cross-validation scores SVC:", cv_scores)
    print("Mean accuracy:", np.mean(cv_scores))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label='PD')
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve SVC")
    plt.legend()
    plt.show()

    print(clf.predict_proba(X_test))




#svc(X_train, X_test, y_train, y_test, 'rbf')
rf(X_train, X_test, y_train, y_test)



# print(y)
# print(X.head())