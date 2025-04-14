import pickle

with open("xgboost_sheet_1", 'rb') as file:
    best_models = pickle.load(file)

# get the best model
best_model = best_models[0]

# Bayes optimization object
opt = best_model[1]

print("Best Model Feature Importances")
importances = opt.best_estimator_.steps[0][1].feature_importances_
named_importances = list(zip(best_model[3], importances))
sorted_feature_importances = sorted(named_importances, key=lambda item: item[1], reverse=True)
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance}")

with open("best_features_model", 'rb') as file:
    best_model = pickle.load(file)

# Bayes optimization object
opt = best_model[1]

print("Top Features Model Feature Importances")
importances = opt.best_estimator_.steps[0][1].feature_importances_
named_importances = list(zip(best_model[3], importances))
sorted_feature_importances = sorted(named_importances, key=lambda item: item[1], reverse=True)
for feature, importance in sorted_feature_importances:
    print(f"{feature}: {importance}")