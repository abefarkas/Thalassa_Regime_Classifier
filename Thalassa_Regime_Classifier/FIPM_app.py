import streamlit as st
import numpy as np # np mean, np random
import pandas as pd
import time # to simulate a real time data, time loop
import plotly.express as px # interactive charts
import plotly.graph_objects as go





st.set_page_config(
    page_title = 'Real-Time Thalassa Regime Classifier',
    page_icon = '🌊',
    layout = 'wide'
)

# dashboard title

st.title("Real-Time / Thalassa Regime Classifier")

# top-level filters

#job_filter = st.selectbox("Select the Job", pd.unique(df['job']))


# creating a single-element container.
placeholder = st.empty()

# dataframe filter

#df = df[df['job']==job_filter]

# near real-time / live feed simulation

for seconds in range(200):
    # read csv
    df = pd.read_csv("predicted_values.csv")
#while True:

    
    with placeholder.container():
        # create three columns
        #kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        #kpi1.metric(label="Age ⏳", value=round(avg_age), delta= round(avg_age) - 10)
        #kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
        #kpi3.metric(label="A/C Balance ＄", value= f"$ {round(balance,2)} ", delta= - round(balance/count_married) * 100)

        # create two columns for charts

        fig_col1, = st.columns(1)
        with fig_col1:
            st.markdown("### Gauge")
            
            
            
            bids = list(df[df.columns[pd.Series(df.columns).str.startswith('bs')][::-1]].tail(1).values[0])
            bids = list([np.nan]*len(bids))+bids
                        
            asks = list(df[df.columns[pd.Series(df.columns).str.startswith('as')]].tail(1).values[0])
            asks = asks+list([np.nan]*len(asks))
            
            bids_price = list(df[df.columns[pd.Series(df.columns).str.startswith('bp')][::-1]].tail(1).values[0])
            asks_price = list(df[df.columns[pd.Series(df.columns).str.startswith('ap')]].tail(1).values[0])
            x = bids_price + asks_price
                        
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=bids, mode='lines', fill='tozeroy', fillcolor='rgba(0,0,255,.75)')) # fill down to xaxis
            fig.add_trace(go.Scatter(x=x, y=asks, mode='lines', fill='tozeroy', fillcolor='rgba(255,0,0,.4)')) # fill to trace0 y
            st.write(fig)
        # with fig_col2:
        #     st.markdown("### Live Predictions")
        #     fig2 = px.histogram(data_frame = df, x = 'age_new')
        #     st.write(fig2)
        #st.markdown("### Detailed Data View")
        #st.dataframe(df)
        time.sleep(1)
    #placeholder.empty()


