import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import umap
import matplotlib.pyplot as plt
import keyboard

# data = pd.read_csv("PPMI_Cohort_Filtered.csv")

# y = y = data["PPMI_COHORT"].values
# X = data.iloc[:, 3:]

# Create progression data set
data = pd.read_csv("data.csv")
patient_data = pd.read_csv("patient_data.csv")

patient_data = patient_data[patient_data["PPMI_CLINICAL_EVENT"]!="NOT_APPLICABLE"]
patient_IDs = patient_data["PPMI_PATNO"].unique()

sample_dict = {}
for ID in patient_IDs:
    df = patient_data[patient_data["PPMI_PATNO"] == ID].reset_index()
    if df.loc[0,'PPMI_CLINICAL_EVENT'] == "BL":
        sample_dict[ID] = [[df.loc[0,'PARENT_SAMPLE_NAME'],df.loc[1,'PARENT_SAMPLE_NAME']],df.loc[0,'PPMI_COHORT']]
    else:
        sample_dict[ID] = [[df.loc[1,'PARENT_SAMPLE_NAME'],df.loc[0,'PARENT_SAMPLE_NAME']],df.loc[0,'PPMI_COHORT']]

new_columns = data.columns.tolist()
new_columns.remove('PARENT_SAMPLE_NAME')
new_columns.remove('Unnamed: 0')
new_columns.insert(0,"PPMI_COHORT")


new_df = pd.DataFrame(columns=new_columns)

row_num = 0
for ID in patient_IDs:
    for column in new_df.columns:
        if column == "PPMI_COHORT":
            new_df.loc[row_num, "PPMI_COHORT"] = sample_dict[ID][1]
        if column in data.columns:
            # Normalized difference
            bl_value = float(data.loc[data["PARENT_SAMPLE_NAME"]==sample_dict[ID][0][0],column].values[0])
            v06_value = float(data.loc[data["PARENT_SAMPLE_NAME"]==sample_dict[ID][0][1],column].values[0])
            new_df.loc[row_num, column] = (v06_value-bl_value)
          
    row_num+=1
for column in new_df.columns:
    if column != "PPMI_COHORT":
        col_range = new_df[column].max()-new_df[column].min()
        if col_range != 0:
            new_df[column] = new_df[column]/col_range
        else:
            new_df[column] = new_df[column]*0


zero_percentage = (new_df == 0).sum() / len(new_df)
columns_to_drop = zero_percentage[zero_percentage >= 0.75].index
progression_df = new_df.drop(columns=columns_to_drop)

y = new_df["PPMI_COHORT"].values
X = new_df.iloc[:, 2:]

class Multi_Plot_2d:
    def __init__(self, rows, columns):
   
        self.fig, axes = plt.subplots(rows, columns, figsize=(20,10))    
        self.axes = axes.flatten()
    
    def show_plot(self):
        self.fig.tight_layout()
        self.fig.show()
    
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
                        c=[sns.color_palette()[x] for x in new_df.PPMI_COHORT.map({"PD":0, "Control":1})])
                        self.axes[plot_num].set_title("N_Neighbor:{},Min_Dist:{},n_components:{},metric:{}".format(neighbor,dist,component,metric), fontsize=6)
                        plot_num +=1

# plot1 = Multi_Plot_2d(4,4)
# plot1.UMAP(X, neighbors=[2, 5, 10, 20],min_dist=[0, 0.2, 0.5, 0.99],n_components=[2],metrics=["euclidean"])
# plot1.show_plot()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33)

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0, class_weight={"Control":1,"PD":1})
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_train)
# y_predt = clf.predict(X_test)
# from sklearn.metrics import confusion_matrix
# print("Depth = {}".format(clf.get_depth()))
# print(confusion_matrix(y_train, y_pred)) 
# print(confusion_matrix(y_test, y_predt))

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

# # Naive Bayes... just for fun.
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_train)
# y_predt = gnb.predict(X_test)

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_train, y_pred)) 
# print(confusion_matrix(y_test, y_predt))

print("Done!")

