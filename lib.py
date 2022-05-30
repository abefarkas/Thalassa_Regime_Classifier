import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class FinancialFeatures(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass




    def WAP(self, X):
        X['WAP'] = (X['bp1']*X['bs1']
                +X['bp2']*X['bs2']
                +X['ap1']*X['as1']
                +X['ap2']*X['as2'])/(X['bs1']+
                                            X['bs2']+
                                            X['as1']+
                                            X['as2'])
        return X

    def build_df(self, X):
        return X

    def spread(self, X):
        X['spread'] = ((X['ap1']/X['bp1']) - 1)
        return X

    def log_price(self, X):
        X['log_price'] = np.log(X['WAP'])
        return X

    def log_returns(self, X):
        X['log_returns'] = X.log_price.diff()
        return X

    def volatility_df(self, X):
        X['realized_volatility'] = np.std(X.log_returns)
        return X

    def volatility_next_period(self, X):
        X['volatility_t+1'] = X['realized_volatility'].shift(-1)
        return X

    def dropping_columns(self, X):
        X.drop(['Unnamed: 0'], axis = 1, inplace = True)
        X.drop(['realized_volatility'], axis = 1, inplace = True)
        return X

    def first2_bid_depth(self, X):
        X['first2_bid_depth'] = X[['bs1', 'bs2']].sum(axis=1)
        return X

    def first2_ask_depth(self, X):
        X['first2_ask_depth'] = X[['as1', 'as2']].sum(axis=1)
        return X

    def full_bid_depth(self, X):
        X['full_bid_depth'] = X[['bs1', 'bs2', 'bs3','bs4', 'bs5', 'bs6','bs7', 'bs8', 'bs9','bs10',
                            'bs11', 'bs12', 'bs13','bs14', 'bs15', 'bs16','bs17', 'bs18', 'bs19','bs20']].sum(axis=1)
        return X

    def full_ask_depth(self, X):
        X['full_ask_depth'] = X[['as1', 'as2', 'as3','as4', 'as5', 'as6','as7', 'as8', 'as9','as10',
                            'as11', 'as12', 'as13','as14', 'as15', 'as16','as17', 'as18', 'as19','as20']].sum(axis=1)
        return X

    def BBAOFI(self, X):
        X['BBAOFI'] = (X['bs1']-X['as1'])/(X['bs1']+X['as1'])
        return X

    def first2_OFI(self, X):
        X['First2OFI'] = ((X['bs1']+X['bs2']) - (X['as1']+X['as2']))/ ((X['bs1']+X['bs2']) + (X['as1']+X['as2']))
        return X

    def FDOFI(self, X):
        X['FDOFI'] = (X['full_bid_depth']-X['full_ask_depth'])/(X['full_bid_depth']+X['full_ask_depth'])
        return X

    def set_index(self, X):
        X.set_index('primary_key')
        return X

    def WPA_trend(self, X):
        X['WAP_trend5'] = X['WAP'].ewm(span=2).mean()
        X['WAP_trend10'] = X['WAP'].ewm(span=5).mean()
        X['WAP_trend20'] = X['WAP'].ewm(span=10).mean()
        X['WAP_trend50'] = X['WAP'].ewm(span=20).mean()
        X['WAP_trend100'] = X['WAP'].ewm(span=50).mean()
        X['WAP_trend200'] = X['WAP'].ewm(span=100).mean()
        X['WAP_trend1000'] = X['WAP'].ewm(span=200).mean()
        return X

    def first2_OFI_trend(self, X):
        X['First2OFI_trend5'] = X['First2OFI'].ewm(span=2).mean()
        X['First2OFI_trend10'] = X['First2OFI'].ewm(span=5).mean()
        X['First2OFI_trend20'] = X['First2OFI'].ewm(span=10).mean()
        X['First2OFI_trend50'] = X['First2OFI'].ewm(span=20).mean()
        X['First2OFI_trend100'] = X['First2OFI'].ewm(span=50).mean()
        X['First2OFI_trend200'] = X['First2OFI'].ewm(span=100).mean()
        X['First2OFI_trend1000'] = X['First2OFI'].ewm(span=200).mean()
        return X

    def FDOFI_trend(self, X):
        X['FDOFI_trend5'] = X['FDOFI'].ewm(span=2).mean()
        X['FDOFI_trend10'] = X['FDOFI'].ewm(span=5).mean()
        X['FDOFI_trend20'] = X['FDOFI'].ewm(span=10).mean()
        X['FDOFI_trend50'] = X['FDOFI'].ewm(span=20).mean()
        X['FDOFI_trend100'] = X['FDOFI'].ewm(span=50).mean()
        X['FDOFI_trend200'] = X['FDOFI'].ewm(span=100).mean()
        X['FDOFI_trend1000'] = X['FDOFI'].ewm(span=200).mean()
        return X

    def fit(self, X, y=None):
        # Store here what needs to be stored during .fit(X_train) as instance attributes.
        # Return "self" to allow chaining .fit().transform()
        return self


    def transform(self, X, y=None):
        # df = FinancialFeatures('/Users/marcostellez/code/abefarkas/Thalassa_Regime_Classifier/raw_data/data_set.csv')
        self.WAP(X)
        self.spread(X)
        self.log_price(X)
        self.log_returns(X)
        self.volatility_df(X)
        self.volatility_next_period(X)
        self.dropping_columns(X)
        self.first2_bid_depth(X)
        self.first2_ask_depth(X)
        self.full_bid_depth(X)
        self.full_ask_depth(X)
        self.BBAOFI(X)
        self.first2_OFI(X)
        self.FDOFI(X)
        self.set_index(X)
        self.WPA_trend(X)
        self.first2_OFI_trend(X)
        self.FDOFI_trend(X)
        return self.build_df(X)



def main():
    df = FinancialFeatures('/Users/marcostellez/code/abefarkas/Thalassa_Regime_Classifier/raw_data/data_set.csv')
    df.WAP()
    df.spread()
    df.log_price()
    df.log_returns()
    df.volatility_df()
    df.volatility_next_period()
    df.dropping_columns()
    df.first2_bid_depth()
    df.first2_ask_depth()
    df.full_bid_depth()
    df.full_ask_depth()
    df.BBAOFI()
    df.first2_OFI()
    df.FDOFI()
    df.set_index()
    df.WPA_trend()
    df.first2_OFI_trend()
    df.FDOFI_trend()
    df = df.build_df()

    return df
if __name__ == "__main__":

   print(main())
