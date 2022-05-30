from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from lib import FinancialFeatures
from statsmodels.tsa.arima.model import ARIMA

class GetStuffDone():
    def __init__(self):
        pass


    def pipe(self):
        self.preproc = Pipeline([
            ('FinFeatures', FinancialFeatures()),
            ('Imputer', SimpleImputer(strategy = 'mean')),
            ('scaler', MinMaxScaler()),
            'model', ARIMA(order=(2, 1,3), trend='t')])
        return self
