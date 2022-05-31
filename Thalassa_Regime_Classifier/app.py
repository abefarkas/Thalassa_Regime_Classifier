#from asyncio.windows_utils import pipe
from matplotlib.axis import XAxis
from matplotlib.pyplot import title
import streamlit as st
import numpy as np # np mean, np random
import pandas as pd
import time # to simulate a real time data, time loop
import plotly.express as px
import plotly.graph_objects as go
import joblib
from lib import FinancialFeatures

model = joblib.load('../fitted_model.joblib')
pipeline = joblib.load('../pipeline.joblib')
financial_features = FinancialFeatures()

st.set_page_config(
    page_title = 'Thalassa app',
    page_icon = '🌊',
    layout = 'wide'
)

# dashboard title
st.title("🌊 Thalassa")
st.write('''
         This app helps decision-making by predicting the volability regime of the cryptomarket in real-time.
         ''')

# creating two single-element container
placeholder1 = st.empty()
placeholder2 = st.empty()

# near real-time / live feed simulation

for seconds in range(500):
    df = pd.read_csv("predicted_values.csv")
    #df = financial_features.transform(df)
    #df = pipeline.transform(df)

    # if ARIMA model:
    #model.append(df['volatility_t+1'])
    #prediction = model.forecast()

    # if another model:
    ## model.predict(...)

    with placeholder1.container():

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Volatility Gauge")
            fig1 = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = df['bs1'].iat[-1],
                mode = "gauge+number+delta",
                delta = {'reference': 1}, # put here the right delta (high vs low volatility)
                title = {'text': "Predicted volatility"},
                gauge = {'axis': {'range': [None, 20]},
                        'steps' : [
                            {'range': [0, 9.999999], 'color': "lightgray"},
                            {'range': [10,20], 'color': "gray"}]}))
            st.write(fig1)

        with fig_col2:
            st.markdown("### Depth Chart")

            bids = list(df[df.columns[pd.Series(df.columns).str.startswith('bs')][::-1]].tail(1).values[0])
            bids = bids+list([np.nan]*len(bids))
            #print('BIDS', bids)
            #print('----------------------------------')

            asks = list(df[df.columns[pd.Series(df.columns).str.startswith('as')]].tail(1).values[0])
            asks = list([np.nan]*len(asks))+asks

            bids_price = list(df[df.columns[pd.Series(df.columns).str.startswith('bp')][::-1]].tail(1).values[0])
            asks_price = list(df[df.columns[pd.Series(df.columns).str.startswith('ap')]].tail(1).values[0])
            x = bids_price + asks_price
            #print('BIDS_PRICE', bids_price)
            #print('----------------------------------')

            fig2 = go.Figure()
            # bids should be on the left side in blue, and aks on the right in red, U-shape
            fig2.add_trace(go.Scatter(x=x, y=bids, mode='lines', fill='tozeroy', name='Bids')) # fill down to xaxis
            fig2.add_trace(go.Scatter(x=x, y=asks, mode='lines', fill='tozeroy', name='Asks')) # fill to trace0 y
            fig2.update_xaxes(title='Prices')
            fig2.update_yaxes(title='Size', range=(0,25))
            st.write(fig2)

        #st.dataframe(df.head(5))

    #with placeholder2.container():

        #fig_col3, fig_col4 = st.columns(2)
        #with fig_col3:
        #st.markdown("### Predictions")
        #    fig3 = px.line(data_frame = df, x = df['primary_key'], y = df['bp1'])
        #    st.write(fig3)

        #with fig_col4:
        #    st.markdown("### Predictions")
        #    fig4 = px.histogram(data_frame = df, x = df['bs3'])
        #    st.write(fig4)

        time.sleep(1)
    #placeholder.empty()
