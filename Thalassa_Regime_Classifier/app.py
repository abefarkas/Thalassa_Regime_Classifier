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

# creating a single-element container
placeholder = st.empty()

# near real-time / live feed simulation

for seconds in range(200):
    df = pd.read_csv("df_clean.csv")

    with placeholder.container():

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 250,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Volatility"}))
            st.write(fig)

        with fig_col2:
            st.markdown("### Live Predictions")
            fig2 = px.scatter(data_frame = df, x = df['primary_key'], y = df['bp1'])
            st.write(fig2)

        time.sleep(1)
    #placeholder.empty()
