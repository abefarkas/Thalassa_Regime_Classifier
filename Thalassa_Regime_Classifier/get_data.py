
# # Imports here !!!
from importlib.resources import path
import os
import pandas as pd
from datetime import datetime, timezone

def get_data(path_to_csv_file):

    # getting data
    df_ob = pd.read_csv(path_to_csv_file)

    # from unix timestamp to human date
    unix_timestamp = lambda x: datetime.fromtimestamp(x/1000.0, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_ob['primary_key']=df_ob['ts'].apply(unix_timestamp).apply(str2date)

    # aggregating by seconds and creating primary key
    # dropping useless columns
    df_agg = df_ob.groupby(pd.Grouper(key='primary_key', axis=0, freq='S')).mean()
    df_agg.reset_index(inplace=True)
    date2str = lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S')
    df_agg['primary_key'] = df_agg['primary_key'].apply(date2str)
    
    df_agg.drop(columns=['last_update_id','ts'], inplace=True)
    
    return df_agg

if __name__=="__main__":
    
    path_to_file = '/Users/fipm/code/abefarkas/Thalassa_Regime_Classifier/raw_data/BTCUSDT_S_DEPTH_20220519.csv'
    data = get_data(path_to_file)
    data.to_csv('/Users/fipm/code/abefarkas/Thalassa_Regime_Classifier/raw_data/data_set_v2.csv',
                index=False)
