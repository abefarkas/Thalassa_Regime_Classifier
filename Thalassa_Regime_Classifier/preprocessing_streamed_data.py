'''Imports here'''
import os
import pandas as pd
from datetime import datetime, timezone

def preprocessing_streamed_data(df_ob, rolling_window=3):
    '''preprocessing of data for streamed data'''
    
    # from unix timestamp to human date
    unix_timestamp = lambda x: datetime.fromtimestamp(x/1000.0, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_ob['timestamp']=df_ob['ts'].apply(unix_timestamp).apply(str2date)

    # aggregating by seconds     
    df_agg = df_ob.groupby(pd.Grouper(key='timestamp', axis=0, freq='S')).mean()
    # applying rolling window of rolling_window lenght
    df_agg = df_agg.rolling(str(rolling_window)+'S').mean()
    df_agg.reset_index(inplace=True)
    # dropping useless columns
    df_agg.drop(columns=['last_update_id','ts'], inplace=True)

    return df_agg

if __name__ == '__main__':
    path_to_raw_data = os.path.dirname(os.getcwd())
    path_to_csv_file = os.path.join(path_to_raw_data,'raw_data','BTCUSDT_S_DEPTH_20220519.csv')
    print(path_to_csv_file)
    df_ob = pd.read_csv(path_to_csv_file)
    db = preprocessing_streamed_data(df_ob, rolling_window=3)
    print(db.head(5))
    