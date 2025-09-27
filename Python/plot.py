import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from CSFData import getter
import utility_functions as uf

class model_analyzer:
    def __init__(self, picklepath, modelname="model"):
        """
        Initializes the model_analyzer class by loading a saved model and preparing datasets.

        Parameters:
            picklepath: str
                Path to the pickle file containing saved model information.
            modelname: str, optional
                A name for the model to be used in plot titles and labels. Default is "model".
        """

        config = uf.get_config()
        logger = uf.Logger()

        self.kwargs = {
            'config': config,
            'logger': logger
        }

        self.picklepath = picklepath
        self.name = modelname
        self.load_best()

    def load_best(self):
        """
        Loads the best model from a pickle file and prepares training/test datasets.

        Returns:
            None — stores baseline and follow-up datasets as instance variables:
                self.BL: dict containing train-test split from baseline data
                self.V06: dict containing follow-up (V06) data
                self.X_cols_best: list of selected features used by the model
        """
        # === Load Best Model ===
        with open(self.picklepath, 'rb') as file:
            best_models = pickle.load(file)
        best_model_tuple = best_models[0]
        self.opt_best = best_model_tuple[1]
        self.X_cols_best = best_model_tuple[3]
        best_thresh = best_model_tuple[2]

        # Get the baseline data, then split
        data_best = getter(datasheet=1, 
                          group='BL',
                          **self.kwargs
                          )
        X_best, y_best = data_best.getXy_selectfeatures(columns=self.X_cols_best)
        X_train, X_test, y_train, y_test = train_test_split(X_best, y_best, test_size=0.2, stratify=y_best, random_state=42)
        self.BL = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
        # Get the V06 data. No split because it is has not been used for training
        data_best_v06 = getter(datasheet=1, 
                          group='V06',
                          **self.kwargs
                          )
        X_v06, y_v06 = data_best_v06.getXy_selectfeatures(columns=self.X_cols_best)
        self.V06 = {
            "X": X_v06,
            "y": y_v06
        }

    # === add AUC Curves ===
    def add_roc(self, X_test, y_test, label, color):
        """
        Adds a ROC curve to the current plot using predicted probabilities from the model.

        Parameters:
            X_test: pandas DataFrame
                Test feature data.
            y_test: array-like
                True labels corresponding to X_test.
            label: str
                Label for the curve in the plot legend.
            color: str
                Color of the curve to be plotted.

        Returns:
            None — directly plots the ROC curve on the active figure.
        """

        y_proba = self.opt_best.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=label+" (area = %0.2f)" % roc_auc)
        return 

    def plot_roc(self):
        """
        Plots ROC curves comparing baseline and follow-up datasets using the best model.

        Returns:
            None — displays the ROC plot.
        """

        plt.figure(figsize=(10, 6))
        
        self.add_roc(
            self.BL["X_test"], 
            self.BL["y_test"], 
            label=f"{self.name} BL", 
            color = 'darkorange')
        
        self.add_roc(
            self.V06["X"], 
            self.V06["y"], 
            label=f"{self.name} Follow-up", 
            color = 'blue')
        
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.name}: Baseline vs. Follow-up')
        plt.legend(loc="lower right")
        plt.savefig(f'Images/roc/{self.name}')
        # plt.show()

    def plot_importances(self):
        """
        Plots a horizontal bar chart showing the most important features in the model.

        Returns:
            None — displays a matplotlib bar chart using chemical names in place of IDs.
    
        Note:
            This function uses a lookup from getter().getChemNames() to convert feature IDs
            to human-readable chemical names. Falls back to IDs if names are unavailable.
        """

        chem_df = getter(datasheet=1, 
                          group='BL',
                          **self.kwargs
                          ).getChemNames()
    
        # Create lookup dictionary: {CHEM_ID: CHEMICAL_NAME}
        id_to_name = chem_df['CHEMICAL_NAME'].to_dict()

        # Hardcode patient sex as a 'chemical name' for feature importance
        id_to_name['PPMI_SEX'] = 'patient sex'

        # Get feature importances
        importances = self.opt_best.best_estimator_.steps[0][1].feature_importances_
        named_importances = list(zip(self.X_cols_best, importances))

        # Optionally map feature names using lookup
        def safe_lookup(fid):
            try:
                return id_to_name.get(int(fid), fid)
            except (ValueError, TypeError):
                return fid

        mapped_named_importances = [
            (safe_lookup(fid), imp) for fid, imp in named_importances
]

        # Sort and select top 15
        sorted_importances = sorted(mapped_named_importances, key=lambda x: x[1], reverse=True)[:15]
        features, import_vals = zip(*sorted_importances)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(features[::-1], import_vals[::-1])
        plt.xlabel("Importance in model")
        plt.title(f"Metabolite importance in {self.name}")
        plt.tight_layout()
        plt.savefig(f'Images/importance/{self.name}')
        # plt.show()

    def plot_confusion_matrix(self, dataset="BL"):
        """
        Generates and displays a confusion matrix for model predictions on a chosen dataset.

        Parameters:
            dataset: str, optional
                Which dataset to evaluate. Use "BL" for baseline (default), "V06" for follow-up.

        Returns:
            None — prints evaluation metrics and displays a confusion matrix plot.
        """

        X, y = None, None
        if (dataset=="BL"):
            X = self.BL["X_test"]
            y = self.BL["y_test"]
        else:
            X = self.V06["X"]
            y = self.V06["y"]
        
        le = getter(datasheet=1, 
                        group='V06',
                        **self.kwargs
                        )

        predictions = self.opt_best.predict(X)
        
        cm = confusion_matrix(y, predictions)
        tn, fp, fn, tp = cm.ravel()
        print("tn is {}, tp is {}, fn is {}, fp is {}".format(tn, tp, fn, fp))
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        accuracy = (tp+tn)/(tn+tp+fn+fp)
        print("Specificity is {}, Sensitivity is {}, Accuracy is {}".format(specificity, sensitivity, accuracy))

        predictions = le.unencode(predictions)
        y = le.unencode(y)

        disp = ConfusionMatrixDisplay.from_predictions(y, predictions)
        plt.title(f"Confusion Matrix {self.name}: {dataset}")
        plt.savefig(f'Images/confusion/{dataset} {self.name}')
        # plt.show()

if __name__ == '__main__':

    analyzer = model_analyzer("Python\picklejar\RF with Original Metabolites", "RF with Original Metabolites")

    analyzer.plot_roc()
    analyzer.plot_importances()
    analyzer.plot_confusion_matrix("BL")
    analyzer.plot_confusion_matrix("V06")

