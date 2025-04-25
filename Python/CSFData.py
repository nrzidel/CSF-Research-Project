import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest
from sklearn.impute import KNNImputer

class getter:
    def __init__(self, datasheet=1, group = "BL"):
        data = self.getdata(datasheet=datasheet, group=group)
        data.columns = data.columns.astype(str)

        self.y = data["PPMI_COHORT"]
        self.X = data.drop(data.columns[0:1], axis=1)
        self.le = LabelEncoder()
    
    def getdata(self, path=".\Data\FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", datasheet=3, group = "BL"):
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
        # print(path)
        patient_data = pd.read_excel(
            path,
            sheet_name = "Sample Meta Data",
            header=0,
            usecols = ["PARENT_SAMPLE_NAME", "COHORT", "PPMI_CLINICAL_EVENT", "PPMI_COHORT"],
            index_col="PARENT_SAMPLE_NAME"
        )
        patient_data = patient_data.drop(patient_data[
            (patient_data.COHORT != "PPMI") |
            (patient_data.PPMI_CLINICAL_EVENT != group)
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
    
    def cleanData(self, nathresh=.5):
        """
        cleanData modifies Model.X to remove columns containing more than nathresh % missing values. 
        Any missing values that are not removed are imputed using K Nearest Neigbors.

        Parameters:
            nathresh: Double Default = .5 (50%); threshold of missing values permitted before a column is dropped
        """
        # % of samples allowed to be NA before column is dropped
        self.X = self.X.dropna(axis=1, thresh=int((1 - nathresh) * self.X.shape[0]))

    
    def getXy(self, nathresh=.5):
        """
        getXy will call cleanData and return X and y dataframes

        Parameters:
            nathresh: Double Default = .5 (50%); threshold of missing values permitted before a column is dropped

        Returns:
            X: pandas dataframe containing kselect features.
            y: array-like containing the class values for the PPMI dataset.
        """
        self.y = self.le.fit_transform(self.y)

        self.cleanData(nathresh=nathresh)

        return self.X, self.y

    def get_X_columns(self):
        return self.X.columns
    
    def getXy_selectfeatures(self, columns = None):   
        self.y = self.le.fit_transform(self.y)
        
        self.X = self.X[columns]
        return self.X, self.y
    
    def unencode(self, y):
        self.le.fit_transform(self.y)
        return self.le.inverse_transform(y)

    def getChemNames(self, path=".\Data\FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX"):
        chem_data = pd.read_excel(
            path,
            sheet_name = "Chemical Annotation",
            header=0,
            usecols = ["CHEM_ID", "CHEMICAL_NAME"],
            index_col="CHEM_ID"
        )
        return chem_data
