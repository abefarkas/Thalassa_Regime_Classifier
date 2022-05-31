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
from data_model_flow_sarah import DataModelPipeline #TO CHANGE

# ------------------------------------------------------------------------------

# instanciate the data-model-flow class
data_model_pipeline = DataModelPipeline()
arima_fitted = joblib.load('../arima_fitted.joblib')

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
         ''')

# creating two single-element container
placeholder1 = st.empty()
placeholder2 = st.empty()

# near real-time / live feed simulation

for seconds in range(500):
    data = pd.read_csv("predicted_values.csv")

    df = data_model_pipeline.financial_features(data)
    y, X = data_model_pipeline.pipeline(df)
    predictions = data_model_pipeline.predict(model=arima_fitted, steps=1)

    with placeholder1.container():

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Volatility Gauge")
            fig1 = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = 1000*predictions['realized_volatility'].iloc[-1],
                mode = "gauge+number+delta",
                delta = {'reference': 0.02}, # put here the right delta (high vs low volatility)
                title = {'text': "Predicted volatility"},
                gauge = {'axis': {'range': [None, 0.04]},
                        'steps' : [
                            {'range': [0, 0.02], 'color': "lightgray"},
                            {'range': [0.02,0.04], 'color': "gray"}]}))
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
            fig2.update_yaxes(title='Size', range=(0,10))
            st.write(fig2)


    with placeholder2.container():

        fig_col3, = st.columns(1)
        with fig_col3:
            st.markdown("### Predictions")
            fig3 = px.scatter(data_frame = predictions, x = predictions.index, y = predictions['realized_volatility'])
            st.write(fig3)

        #with fig_col4:
        #    st.markdown("### Predictions")
        #    fig4 = px.histogram(data_frame = df, x = df['bs3'])
        #    st.write(fig4)

        time.sleep(1)
    #placeholder.empty()
