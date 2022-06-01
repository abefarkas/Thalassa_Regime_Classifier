#from asyncio.windows_utils import pipe
from matplotlib.axis import XAxis
from matplotlib.pyplot import title, xlabel, ylabel
import streamlit as st
import numpy as np # np mean, np random
import pandas as pd
import time # to simulate a real time data, time loop
import plotly.express as px
import plotly.graph_objects as go
import joblib
from data_model_flow import DataModelPipeline

# ------------------------------------------------------------------------------

# instanciate the data-model-flow class
data_model_pipeline = DataModelPipeline()
model = joblib.load('../model.joblib')

# ------------------------------------------------------------------------------

st.set_page_config(
    page_title = 'Thalassa',
    page_icon = '🌊',
    layout = 'wide'
)

# dashboard title
st.title("🌊 Thalassa Trading Tool")
st.write('''
         This app helps fast decision-making by predicting the volatility regime of the cryptomarket in real-time.
         It is currently monitoring BTCUSDT Futures on Binance.
         ''')

# creating two single-element container
placeholder1 = st.empty()
placeholder2 = st.empty()

# near real-time / live feed simulation

for seconds in range(500):
    data = pd.read_csv("predicted_values.csv")

    df = data_model_pipeline.financial_features(data)
    y, X = data_model_pipeline.pipeline(df)
    predictions = data_model_pipeline.predict(model=model)
    print(predictions)

    with placeholder1.container():

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Volatility Gauge")
            fig1 = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = predictions['realized_volatility'].iloc[-1],
                mode = "gauge+number",#+delta
                #delta = {'reference': 0.02}, # put here the right delta (high vs low volatility)
                title = {'text': "Predicted volatility (30 seconds from now)"},
                gauge = {'bar': {'color': 'grey'},
                        'axis': {'range': [None, y.max()[0]]},
                        'steps' : [
                            {'range': [0, (y.max()[0]/2)], 'color': '#00CC96'},
                            {'range': [(y.max()[0]/2),y.max()[0]], 'color': '#EF553B'}]}))
            st.write(fig1)

        with fig_col2:
            st.markdown("### Depth Chart")

            bids = list(df[df.columns[pd.Series(df.columns).str.startswith('bs')]].tail(1).values[0])
            bids = list(np.cumsum(bids)[::-1])
            bids = bids+list([np.nan]*len(bids))

            asks = list(df[df.columns[pd.Series(df.columns).str.startswith('as')]].tail(1).values[0])
            asks = list(np.cumsum(asks))
            asks = list([np.nan]*len(asks))+asks

            bids_price = list(df[df.columns[pd.Series(df.columns).str.startswith('bp')][::-1]].tail(1).values[0])
            asks_price = list(df[df.columns[pd.Series(df.columns).str.startswith('ap')]].tail(1).values[0])
            x = bids_price + asks_price

            fig2 = go.Figure()
            # bids should be on the left side in blue, and aks on the right in red, U-shape
            fig2.add_trace(go.Scatter(x=x, y=bids, mode='lines', fill='tozeroy', name='Bids')) # fill down to xaxis
            fig2.add_trace(go.Scatter(x=x, y=asks, mode='lines', fill='tozeroy', name='Asks')) # fill to trace0 y
            fig2.update_xaxes(title='Prices')
            fig2.update_yaxes(title='Size', range=(0,np.max(bids+asks)))
            st.write(fig2)


    with placeholder2.container():

        #fig_col3, = st.columns(1)
        #with fig_col3:
        st.markdown("### Streamed data")
        fig3 = px.line(
            data_frame = y,
            x = y.index,
            y = y['realized_volatility'])

        fig3.update_layout(
            xaxis_title = 'Time',
            yaxis_title = 'Realized volatility'
        )
        st.write(fig3)

        time.sleep(1)
