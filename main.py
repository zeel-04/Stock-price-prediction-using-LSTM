import streamlit as st
import yfinance as yf
import datetime
import os  
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

st.title('Explore your desired stock')

#ask user for share name
stock_name = st.text_input('Enter Code-name of your desired share')

# stock_name = "AAPL"
# stock = yf.Ticker(stock_name)

# st.info(stock.major_holders)
# st.info(stock.institutional_holders)
# st.info(stock.mutualfund_holders)
# st.info(stock.news)
c1,c2 = st.columns(2)

with c1:
    start_date = st.date_input("Enter a start date",datetime.date(2020, 1, 1))
    st.write(start_date)
with c2:
    end_date = st.date_input("Enter a start date",datetime.date(2022, 1, 1))
    st.write(end_date)

if len(stock_name) != 0:
    st.info(f'Curernt processing Share "{stock_name}"')
    #downloaad stock data as per above mentioned dates
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    os.makedirs('Data', exist_ok=True)
    stock_data.to_csv(f'Data/{stock_name}.csv')
    stock_data = pd.read_csv(f'Data/{stock_name}.csv')

    #show stock_data
    st.dataframe(stock_data,use_container_width=True)

    fig = go.Figure(

        layout=go.Layout(
            title = f"{stock_name} Share price chart"
        ),

        data=[go.Candlestick(
            x=stock_data["Date"],
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            increasing_line_color= 'cyan', 
            decreasing_line_color= 'gray',
        )]
    )

    st.plotly_chart(fig,use_container_width=True,theme="streamlit")

    fig2 = go.Figure()
    trace1 = px.line(stock_data,x='Date',y='Close',title='Close trace')
    trace2 = px.line(stock_data,x='Date',y='Open',title='Open trace')
    trace2.update_traces(line_color='gray')
    fig2.add_trace(trace1.data[0])
    fig2.add_trace(trace2.data[0])
    st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    fig2 = go.Figure()
    trace1 = px.line(stock_data['Close'])
    trace2 = px.line(stock_data['Open'])
    trace2.update_traces(line_color='gray')
    fig2.add_trace(trace1.data[0])
    fig2.add_trace(trace2.data[0])
    st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    
