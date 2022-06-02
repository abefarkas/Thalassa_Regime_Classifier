import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone, timedelta
import joblib


class DataModelPipeline():
    def __init__(self):
        self.preprocessing = Pipeline([
            ('Imputer', SimpleImputer(strategy = 'mean')),
            # ('scaler', MinMaxScaler()),
            ])
        self.data = None
        self.y = None
        self.X = None

    def financial_features(self, data):
        self.data = data.copy()

        # unix_timestamp = lambda x: datetime.fromtimestamp(x/1000.0, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        # self.data['primary_key']=self.data['ts'].apply(unix_timestamp).apply(str2date)
        self.data['primary_key']=self.data['primary_key'].apply(str2date)


        # self.data['less30s']=self.data['primary_key'].dt.second<30
        # self.data['primary_key']=self.data['primary_key'].dt.strftime('%Y-%m-%d %H:%M')+self.data['less30s'].apply(lambda x: ':15' if x==True else ':45')
        # self.data.drop(columns=['less30s'], inplace=True)


        # WAP
        self.data['WAP'] = (self.data['bp1']*self.data['bs1']
                +self.data['bp2']*self.data['bs2']
                +self.data['ap1']*self.data['as1']
                +self.data['ap2']*self.data['as2'])/(self.data['bs1']+
                                            self.data['bs2']+
                                            self.data['as1']+
                                            self.data['as2'])

        # log_price
        self.data['log_price'] = 100*np.log(self.data['WAP'])

        # log_returns
        self.data['log_returns'] = self.data.log_price.diff()


        # other features
        self.data['spread'] = ((self.data['ap1']/self.data['bp1']) - 1)
        self.data['spread_sq']=self.data['spread']*self.data['spread']


        self.data['full_bid_depth'] = self.data[['bs1', 'bs2', 'bs3','bs4', 'bs5', 'bs6','bs7', 'bs8', 'bs9','bs10',
                            'bs11', 'bs12', 'bs13','bs14', 'bs15', 'bs16','bs17', 'bs18', 'bs19','bs20']].sum(axis=1)
        self.data['full_ask_depth'] = self.data[['as1', 'as2', 'as3','as4', 'as5', 'as6','as7', 'as8', 'as9','as10',
                            'as11', 'as12', 'as13','as14', 'as15', 'as16','as17', 'as18', 'as19','as20']].sum(axis=1)
        self.data['BBAOFI'] = (self.data['bs1']-self.data['as1'])/(self.data['bs1']+self.data['as1'])
        self.data['First2OFI'] = ((self.data['bs1']+self.data['bs2']) - (self.data['as1']+self.data['as2']))/ ((self.data['bs1']+self.data['bs2']) + (self.data['as1']+self.data['as2']))
        self.data['FDOFI'] = (self.data['full_bid_depth']-self.data['full_ask_depth'])/(self.data['full_bid_depth']+self.data['full_ask_depth'])

        # realized_volatility
        sigma = lambda x: (np.nansum(x**2))**0.5
        y = self.data[['log_returns']]

        rolling=30
        y = y.rolling(rolling).apply(sigma)


        primary_key=self.data['primary_key']

        self.data = self.data.rolling(rolling).mean()
        self.data['realized_volatility']=y.values
        self.data['primary_key']=primary_key

        #####

        # sigma = lambda x: (np.nansum(x['log_returns']**2))**0.5
        # y = self.data[['primary_key','log_returns']].groupby(['primary_key']).apply(sigma)

        # self.data = self.data.groupby(['primary_key']).mean()
        # self.data.reset_index(drop=False, inplace=True)
        # self.data['realized_volatility']=y.values

        return self.data

    def pipeline(self, data):
        data = data.dropna().reset_index(drop=True)
        self.y = data[['primary_key','realized_volatility']].set_index('primary_key')
        #self.y = data[['realized_volatility']]

        X = data.drop(columns=['realized_volatility','primary_key'])
        self.X = pd.DataFrame(self.preprocessing.fit_transform(X), columns=self.preprocessing.get_feature_names_out())
        return self.y, self.X

    def predict(self, model, pca, gaussian_mixture):

        lag = 0

        y = self.y.reset_index(drop=True)

        X_ = pd.concat((
            y.shift(lag).rename(columns={'realized_volatility':'lag_y'}),
            self.X['spread'].shift(lag),
            self.X['spread_sq'].shift(lag),
            self.X['BBAOFI'].shift(lag),
            self.X['FDOFI'].shift(lag)
            ), axis=1).dropna()

        predictions = model.predict(X_.tail(1))
        predictions = pd.DataFrame.from_dict({'realized_volatility':predictions})

        # PCA & GaussianMixture
        new_y = pd.concat((
            predictions.tail(1),
            y.shift(lag).rename(columns={'realized_volatility':'lag_y'}).tail(1).reset_index(drop=True),
            self.X['spread'].shift(lag).tail(1).reset_index(drop=True),
            self.X['spread_sq'].shift(lag).tail(1).reset_index(drop=True),
            self.X['BBAOFI'].shift(lag).tail(1).reset_index(drop=True),
            self.X['FDOFI'].shift(lag).tail(1).reset_index(drop=True)
            ), axis=1)

        X_emb = pca.transform(new_y.values[:,1:])
        X_emb = np.concatenate((new_y.values[:,0].reshape(new_y.shape[0],-1),X_emb),axis=1)
        new_y = pd.DataFrame(X_emb, columns=('volatility', 'predictors'))

        labels = gaussian_mixture.predict(new_y)
        probs = gaussian_mixture.predict_proba(new_y)[:,1] # 0: high volatility, 1: low volatility

        prediction_regimes = pd.concat((new_y,pd.DataFrame({'labels':labels, 'probs':probs})), axis=1).dropna()

        return predictions, prediction_regimes


if __name__=='__main__':
    data = pd.read_csv('../raw_data/data_set_v2.csv')

    # instanciate the data-model-flow class
    data_model_pipeline = DataModelPipeline()
    # construct finantial features
    df = data_model_pipeline.financial_features(data)
    # getting endogenous and exogenous variables to be used
    # to train a model
    y, X = data_model_pipeline.pipeline(df)

    model = joblib.load('../model.joblib')
    pca = joblib.load('../pca.joblib')
    gaussian_mixture = joblib.load('../gaussian_mixture.joblib')

    # predicting a model with new values for endogenous variable
    predictions, regimes = data_model_pipeline.predict(model=model, pca=pca, gaussian_mixture=gaussian_mixture)
