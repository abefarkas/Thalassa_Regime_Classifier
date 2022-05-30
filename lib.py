import numpy as np
import pandas as pd

class FinancialFeatures():
    def __init__(self, data):
        self.data = data




    def WAP(self):
        self.data['WAP'] = (self.data['bp1']*self.data['bs1']
                +self.data['bp2']*self.data['bs2']
                +self.data['ap1']*self.data['as1']
                +self.data['ap2']*self.data['as2'])/(self.data['bs1']+
                                            self.data['bs2']+
                                            self.data['as1']+
                                            self.data['as2'])
        return self.data

    def build_df(self):
        return self.data

    def spread(self):
        self.data['spread'] = ((self.data['ap1']/self.data['bp1']) - 1)
        return self.data

    def log_price(self):
        self.data['log_price'] = np.log(self.data['WAP'])
        return self.data

    def log_returns(self):
        self.data['log_returns'] = self.data.log_price.diff()
        return self.data

    def volatility_df(self):
        self.data['realized_volatility'] = np.std(self.data.log_returns)
        return self.data

    def volatility_next_period(self):
        self.data['volatility_t+1'] = self.data['realized_volatility'].shift(-1)
        return self.data

    def dropping_columns(self):
        self.data.drop(['Unnamed: 0'], axis = 1, inplace = True)
        self.data.drop(['realized_volatility'], axis = 1, inplace = True)
        return self.data

    def first2_bid_depth(self):
        self.data['first2_bid_depth'] = self.data[['bs1', 'bs2']].sum(axis=1)
        return self.data

    def first2_ask_depth(self):
        self.data['first2_ask_depth'] = self.data[['as1', 'as2']].sum(axis=1)
        return self.data

    def full_bid_depth(self):
        self.data['full_bid_depth'] = self.data[['bs1', 'bs2', 'bs3','bs4', 'bs5', 'bs6','bs7', 'bs8', 'bs9','bs10',
                            'bs11', 'bs12', 'bs13','bs14', 'bs15', 'bs16','bs17', 'bs18', 'bs19','bs20']].sum(axis=1)
        return self.data

    def full_ask_depth(self):
        self.data['full_ask_depth'] = self.data[['as1', 'as2', 'as3','as4', 'as5', 'as6','as7', 'as8', 'as9','as10',
                            'as11', 'as12', 'as13','as14', 'as15', 'as16','as17', 'as18', 'as19','as20']].sum(axis=1)
        return self.data

    def BBAOFI(self):
        self.data['BBAOFI'] = (self.data['bs1']-self.data['as1'])/(self.data['bs1']+self.data['as1'])
        return self.data

    def first2_OFI(self):
        self.data['First2OFI'] = ((self.data['bs1']+self.data['bs2']) - (self.data['as1']+self.data['as2']))/ ((self.data['bs1']+self.data['bs2']) + (self.data['as1']+self.data['as2']))
        return self.data

    def FDOFI(self):
        self.data['FDOFI'] = (self.data['full_bid_depth']-self.data['full_ask_depth'])/(self.data['full_bid_depth']+self.data['full_ask_depth'])
        return self.data

    def set_index(self):
        self.data.set_index('primary_key')
        return self.data

    def WPA_trend(self):
        self.data['WAP_trend5'] = self.data['WAP'].ewm(span=2).mean()
        self.data['WAP_trend10'] = self.data['WAP'].ewm(span=5).mean()
        self.data['WAP_trend20'] = self.data['WAP'].ewm(span=10).mean()
        self.data['WAP_trend50'] = self.data['WAP'].ewm(span=20).mean()
        self.data['WAP_trend100'] = self.data['WAP'].ewm(span=50).mean()
        self.data['WAP_trend200'] = self.data['WAP'].ewm(span=100).mean()
        self.data['WAP_trend1000'] = self.data['WAP'].ewm(span=200).mean()
        return self.data

    def first2_OFI_trend(self):
        self.data['First2OFI_trend5'] = self.data['First2OFI'].ewm(span=2).mean()
        self.data['First2OFI_trend10'] = self.data['First2OFI'].ewm(span=5).mean()
        self.data['First2OFI_trend20'] = self.data['First2OFI'].ewm(span=10).mean()
        self.data['First2OFI_trend50'] = self.data['First2OFI'].ewm(span=20).mean()
        self.data['First2OFI_trend100'] = self.data['First2OFI'].ewm(span=50).mean()
        self.data['First2OFI_trend200'] = self.data['First2OFI'].ewm(span=100).mean()
        self.data['First2OFI_trend1000'] = self.data['First2OFI'].ewm(span=200).mean()
        return self.data

    def FDOFI_trend(self):
        self.data['FDOFI_trend5'] = self.data['FDOFI'].ewm(span=2).mean()
        self.data['FDOFI_trend10'] = self.data['FDOFI'].ewm(span=5).mean()
        self.data['FDOFI_trend20'] = self.data['FDOFI'].ewm(span=10).mean()
        self.data['FDOFI_trend50'] = self.data['FDOFI'].ewm(span=20).mean()
        self.data['FDOFI_trend100'] = self.data['FDOFI'].ewm(span=50).mean()
        self.data['FDOFI_trend200'] = self.data['FDOFI'].ewm(span=100).mean()
        self.data['FDOFI_trend1000'] = self.data['FDOFI'].ewm(span=200).mean()
        return self.data


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
