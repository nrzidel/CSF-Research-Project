import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from CSFData import getter


# === Load Best Model ===
with open("xgboost_sheet_1_n_iter_25", 'rb') as file:
    best_models = pickle.load(file)
best_model_tuple = best_models[0]
opt_best = best_model_tuple[1]
X_cols_best = best_model_tuple[3]
data_best = getter(datasheet=1)
X_best, y_best = data_best.getXy_selectfeatures(columns=X_cols_best)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_best, y_best, test_size=0.2, stratify=y_best, random_state=42)
# === Load Top Features Model ===
with open("best_features_model", 'rb') as file:
    top_feat_model = pickle.load(file)
opt_top = top_feat_model[1]
X_cols_top = top_feat_model[3]
data_top = getter(datasheet=1)
X_top, y_top = data_top.getXy_selectfeatures(columns=X_cols_top)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_top, y_top, test_size=0.2, stratify=y_top, random_state=42)

# === AUC Curves ===
def plot_roc(model, X_test, y_test, label, color):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=label+" (area = %0.2f)" % roc_auc)
    return 

# plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
plot_roc(opt_best.best_estimator_, X_test_b, y_test_b, label="Best Optimized Model", color = 'darkorange')
plot_roc(opt_top.best_estimator_, X_test_t, y_test_t, label="Top Features Model", color = 'blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best Optimized Model vs Top Features Model')
plt.legend(loc="lower right")
plt.show()


# === Feature Importance Bar Charts ===
def plot_importances(opt_model, feature_names, title):
    importances = opt_model.best_estimator_.steps[0][1].feature_importances_
    named_importances = list(zip(feature_names, importances))
    sorted_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)[:15]
    features, import_vals = zip(*sorted_importances)
    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], import_vals[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()
# plot_importances(opt_best, X_cols_best, "Best Model Feature Importances")
plot_importances(opt_top, X_cols_top, "Top Features Model Feature Importances")

def plot_confusion_matrix(model, X_test, y_test, title):
    predictions = model.predict(X_test)
    # cm = confusion_matrix(y_test, predictions, labels = model.classes_)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.plot()
    # plt.title(title)
    plt.show()

plot_confusion_matrix(opt_top, X_test_t, y_test_t, "Top Features Model Confusion Matrix")


