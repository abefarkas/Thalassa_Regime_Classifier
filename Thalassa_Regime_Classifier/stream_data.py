import json
import websocket
import datetime
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time

class StreamingData():
    def __init__(self):
        self.depth = 5
        self.df_0 = None
        self.symbol = 'BTCUSDT'
        self.socket = 'wss://stream.binance.com:9443/ws/{}@depth{}'.format(self.symbol.lower(),self.depth)

    def handle_trades(self,json_message):
        df = pd.DataFrame.from_dict(self.my_json(json_message))
        self.df_0 = pd.concat([self.df_0, df], axis=0)
        df_clean = self.my_return().dropna()
        # save to local machine

    def my_return(self):
        return self.preprocessing_streamed_data(self.df_0, rolling_window=2)

    def on_message(self,wsapp,message):
        json_message = json.loads(message)
        self.handle_trades(json_message)

    def on_error(self,wsapp,error):
        print(error)

    def start(self):
        self.df_0 = pd.DataFrame.from_dict(self.my_json_0(self.depth))
        wsapp = websocket.WebSocketApp(self.socket, on_message=self.on_message, on_error=self.on_error)
        wsapp.run_forever()

    def preprocessing_streamed_data(self,df_ob, rolling_window=2):
        '''preprocessing of data for streamed data'''

        # aggregating by seconds
        df_agg = df_ob.groupby(pd.Grouper(key='primary_key', axis=0, freq='S')).mean()
        # applying rolling window of rolling_window lenght
        df_agg = df_agg.rolling(str(rolling_window)+'S').mean()
        # moving the index as a column
        df_agg.reset_index(inplace=True)

        return df_agg

    def my_json(self, json_message):
        size = len(np.array(json_message['bids'])[:,0])
        return {
            **{'primary_key':[datetime.now()]},
            **{'bp'+str(key):[float(value)] for key,value in zip(np.arange(0,size)+1,np.array(json_message['bids'])[:,0])},
            **{'bs'+str(key):[float(value)] for key,value in zip(np.arange(0,size)+1,np.array(json_message['bids'])[:,1])},
            **{'ap'+str(key):[float(value)] for key,value in zip(np.arange(0,size)+1,np.array(json_message['asks'])[:,0])},
            **{'as'+str(key):[float(value)] for key,value in zip(np.arange(0,size)+1,np.array(json_message['asks'])[:,1])}}

    def my_json_0(self, size):
        return {
            **{'primary_key':[datetime.now()]},
            **{'bp'+str(key):[value] for key,value in zip(np.arange(0,size)+1,(np.arange(0,size)+1)*np.nan)},
            **{'bs'+str(key):[value] for key,value in zip(np.arange(0,size)+1,(np.arange(0,size)+1)*np.nan)},
            **{'ap'+str(key):[value] for key,value in zip(np.arange(0,size)+1,(np.arange(0,size)+1)*np.nan)},
            **{'as'+str(key):[value] for key,value in zip(np.arange(0,size)+1,(np.arange(0,size)+1)*np.nan)}}

if __name__=='__main__':
    w = StreamingData()
    w.start()
    w.my_return() # it is never triggered
