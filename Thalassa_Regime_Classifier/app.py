#from asyncio.windows_utils import pipe
from tkinter import font
from turtle import width
from matplotlib.axis import XAxis
from matplotlib.pyplot import legend, title, xlabel, ylabel
import streamlit as st
import numpy as np # np mean, np random
import pandas as pd
import time # to simulate a real time data, time loop
import plotly.express as px
import plotly.graph_objects as go
import joblib
from data_model_flow import DataModelPipeline
#from streamlit_autorefresh import st_autorefresh

# ------------------------------------------------------------------------------

# instanciate the data-model-flow class
data_model_pipeline = DataModelPipeline()
model = joblib.load('model.joblib')
pca = joblib.load('pca.joblib')
gaussian_mixture = joblib.load('gaussian_mixture.joblib')

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

sigma_max = .12

while True:
    try:
        time.sleep(0.5)

        data = pd.read_csv("predicted_values.csv")

        df = data_model_pipeline.financial_features(data)
        y, X = data_model_pipeline.pipeline(df)
        predictions, regimes = data_model_pipeline.predict(model=model, pca=pca, gaussian_mixture=gaussian_mixture)

        with placeholder1.container():

            # create two columns for charts
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.markdown("### Volatility Gauge")

                regime_probability = regimes['probs'].values[0]
                sigma_max = np.max([y.max()[0],sigma_max]) # sigma_max=y.max()[0]
                sigma_probability = predictions['realized_volatility'].iloc[-1]
                cut_off = sigma_probability*2 - sigma_max + 2*(sigma_max - sigma_probability)*regime_probability

                fig1 = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = sigma_probability,
                    mode = "gauge+number",
                    title = {'text': "<br><span style='font-size:0.8em;color:#636EFA'>Predicted volatility (30 sec. from now)</span><br><span style='font-size:0.8em;color:#00CC96'>Low volatility regime</span><br><span style='font-size:0.8em;color:#EF553B'>High volatility regime</span><br>"},
                    name = 'hello',
                    gauge = {
                        'bar': {'color': '#636EFA'},
                        'axis': {'range': [None, sigma_max]},
                        'steps' : [
                            {'range': [0, cut_off], 'color': '#00CC96'},
                            {'range': [cut_off, sigma_max], 'color': '#EF553B'}]}))

                fig1.update_layout(font = {'color': "#636EFA"})

                st.plotly_chart(fig1, use_container_width=True)

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
                fig2.update_layout(legend=dict(font=dict(size=14)))

                st.plotly_chart(fig2, use_container_width=True)


        with placeholder2.container():

            #fig_col3, = st.columns(1)
            #with fig_col3:
            st.markdown("### Historical Volatility")
            fig3 = px.line(
                data_frame = y,
                x = y.index,
                y = y['realized_volatility'])

            fig3.update_layout(
                xaxis_title = 'Time',
                yaxis_title = 'Realized volatility'
                #width=1000
            )
            #st.write(fig3)
            st.plotly_chart(fig3, use_container_width=True)
    except:
        pass
