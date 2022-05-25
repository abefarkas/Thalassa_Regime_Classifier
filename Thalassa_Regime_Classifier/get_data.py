
# # Imports here !!!
from importlib.resources import path
import os
import pandas as pd
from datetime import datetime, timezone

def get_data(csv_file):

    # getting data
    path_to_raw_data = os.path.dirname(os.getcwd())
    df_ob = pd.read_csv(os.path.join(path_to_raw_data,'raw_data',csv_file))

    # from unix timestamp to human date
    unix_timestamp = lambda x: datetime.fromtimestamp(x/1000.0, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df_ob['timestamp']=df_ob['ts'].apply(unix_timestamp).apply(str2date)

    # aggregating by seconds and creating primary key
    # dropping useless columns
    df_agg = df_ob.groupby(pd.Grouper(key='timestamp', axis=0, freq='S')).mean()
    df_agg.reset_index(inplace=True)
    df_agg['less30s']=df_agg['timestamp'].dt.second<30
    df_agg['primary_key']=df_agg['timestamp'].dt.strftime('%Y-%m-%d %H:%M')+df_agg['less30s'].apply(lambda x: ':15' if x==True else ':45')
    df_agg.drop(columns=['timestamp','less30s','last_update_id','ts'], inplace=True)

    # aggregating by 30 seconds
    df_agg = df_agg.groupby(['primary_key']).mean()
    df_agg.reset_index(inplace=True)

    return df_agg

if __name__=="__main__":
    print(get_data('BTCUSDT_S_DEPTH_20220519.csv'))
