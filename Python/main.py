import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import umap
import matplotlib.pyplot as plt
import keyboard

data = pd.read_csv("CSF-Research-Project\Data\PPMI_Cohort_Filtered.csv")
data = data.drop(data.columns[0], axis=1)   #Drop the first column, which is the sequential numbers from R

y = data["PPMI_COHORT"].values
X = data.iloc[:, 3:]

# Visualization
class Multi_Plot_2d:
    def __init__(self, rows, columns):
   
        self.fig, axes = plt.subplots(rows, columns, figsize=(20,10))    
        self.axes = axes.flatten()
    
    def show_plot(self):
        self.fig.tight_layout()
        plt.show(block=True)
    
    def UMAP(self, X=None, neighbors = [15], min_dist = [0.1], n_components = [2], metrics = ["euclidean"]):
        plot_num = 0
        for neighbor in neighbors:
            for dist in min_dist:
                for component in n_components:
                    for metric in metrics:
                        reducer = umap.UMAP(    
                        n_neighbors=neighbor,
                        min_dist=dist,
                        n_components=component,
                        metric=metric)
                        embedding = reducer.fit_transform(X.values)
                        self.axes[plot_num].scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        c=[sns.color_palette()[x] for x in data.PPMI_COHORT.map({"PD":0, "Control":1})])
                        self.axes[plot_num].set_title("N_Neighbor:{},Min_Dist:{},n_components:{},metric:{}".format(neighbor,dist,component,metric), fontsize=6)
                        plot_num +=1
    
class Multi_Plot_3d:
    
    def __init__(self, rows, columns):
        self.fig, axes = plt.subplots(rows, columns, figsize=(20,10), subplot_kw=dict(projection='3d'))
        self.axes = axes.flatten()
    
    def show_plot(self):
        self.fig.tight_layout()
        plt.show(block=True)
    
    def UMAP(self, X=None, neighbors = [15], min_dist = [0.1], n_components = [3], metrics = ["euclidean"]):
        plot_num = 0
        for neighbor in neighbors:
            for dist in min_dist:
                for component in n_components:
                    for metric in metrics:
                        reducer = umap.UMAP(    
                        n_neighbors=neighbor,
                        min_dist=dist,
                        n_components=component,
                        metric=metric)
                        embedding = reducer.fit_transform(X.values)
                        self.axes[plot_num].scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        embedding[:, 2],
                        c=[sns.color_palette()[x] for x in data.PPMI_COHORT.map({"PD":0, "Control":1})])
                        self.axes[plot_num].set_title("N_Neighbor:{},Min_Dist:{},n_components:{},metric:{}".format(neighbor,dist,component,metric), fontsize=6)
                        plot_num +=1




# UMAP Plots

# plot1 = Multi_Plot_2d(4,4)
# plot1.UMAP(X,neighbors=[2, 5, 10, 20],min_dist=[0, 0.2, 0.5, 0.99],n_components=[2],metrics=["euclidean"])
# plot1.show_plot()
# plot2 = Multi_Plot_3d(4,4)
# plot2.UMAP(X,neighbors=[2, 5, 10, 20],min_dist=[0, 0.2, 0.5, 0.99],n_components=[3],metrics=["euclidean"])
# plot2.show_plot()        


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=32)

# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(random_state=0, class_weight={"Control":0.01,"PD":1})
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_train)
# y_predt = clf.predict(X_test)

# from sklearn.metrics import confusion_matrix
# print("Depth = {}".format(clf.get_depth()))
# print(confusion_matrix(y_train, y_pred)) 
# print(confusion_matrix(y_test, y_predt))

# importance = clf.feature_importances_
# importance[importance !=0] = True
# importance = importance.astype(bool)
# importance_X = X.loc[:, importance]

# plot1 = Multi_Plot_2d(4,4)
# plot1.UMAP(X=importance_X, neighbors=[2, 5, 10, 20],min_dist=[0, 0.2, 0.5, 0.99],n_components=[2],metrics=["euclidean"])
# plot1.show_plot()

from sklearn import svm
from sklearn.metrics import confusion_matrix
# clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
C=[10.0, 1e+02, 1e+03, 1e+04, 1e+05]
gamma=[1e-3, 1e-4, 1e-05, 1e-06, 1e-07]
kernels = ['rbf','linear','poly','sigmoid']
for c in C:
  for g in gamma:
    for kernel in kernels:
        clf = svm.SVC(C=c, gamma=g, kernel=kernel, probability=True) # linear, rbf, polynomial, sigmoid
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(c, g, kernel, (tp+tn)/(tp+tn+fp+fn))

print("Done!")
# print("Press space to continue...")
# keyboard.wait("space")
# print("Script Closed!")

        
 
