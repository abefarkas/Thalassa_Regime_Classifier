import streamlit as st
import numpy as np # np mean, np random
import pandas as pd
import time # to simulate a real time data, time loop
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title = 'Real-Time Thalassa Regime Classifier',
    page_icon = '🌊',
    layout = 'wide'
)

# dashboard title
st.title("Real-Time / Thalassa Regime Classifier")

# creating two single-element container
placeholder1 = st.empty()
placeholder2 = st.empty()

# near real-time / live feed simulation

for seconds in range(500):
    df = pd.read_csv("predicted_values.csv")

    with placeholder1.container():

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Volatility gauge")
            fig1 = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = df['bs1'].iat[-1],
                mode = "gauge+number",
                gauge = {'axis': {'range': [None, 20]},
                        'steps' : [
                            {'range': [0, 9.999999], 'color': "lightgray"},
                            {'range': [10,20], 'color': "gray"}]}))
            st.write(fig1)

        with fig_col2:
            st.markdown("### Predictions")
            fig2 = px.line(data_frame = df, x = df['primary_key'], y = df['bp1'])
            st.write(fig2)

    with placeholder2.container():

        #fig_col3, fig_col4 = st.columns(2)
        #with fig_col3:
        st.markdown("### Bids and asks")

        bids = list(df[df.columns[pd.Series(df.columns).str.startswith('bs')][::-1]].tail(1).values[0])
        bids = list([np.nan]*len(bids))+bids

        asks = list(df[df.columns[pd.Series(df.columns).str.startswith('as')]].tail(1).values[0])
        asks = asks+list([np.nan]*len(asks))

        bids_price = list(df[df.columns[pd.Series(df.columns).str.startswith('bp')][::-1]].tail(1).values[0])
        asks_price = list(df[df.columns[pd.Series(df.columns).str.startswith('ap')]].tail(1).values[0])
        x = bids_price + asks_price

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=bids, mode='lines', fill='tozeroy', fillcolor='rgba(0,0,255,.75)')) # fill down to xaxis
        fig3.add_trace(go.Scatter(x=x, y=asks, mode='lines', fill='tozeroy', fillcolor='rgba(255,0,0,.4)')) # fill to trace0 y
        st.write(fig3)

        #with fig_col4:
        #    st.markdown("### Predictions")
        #    fig4 = px.histogram(data_frame = df, x = df['bs3'])
        #    st.write(fig4)

        time.sleep(1)
    #placeholder.empty()
