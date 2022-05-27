from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from lib import FinancialFeatures

class GetStuffDone():
    def __init__(self):
        pass


    def pipe(self):
        self.preproc = Pipeline([
            ('FinFeatures', FinancialFeatures()),
            ('Imputer', SimpleImputer(strategy = 'mean')),
            ('scaler', MinMaxScaler()),
            'model', ])
