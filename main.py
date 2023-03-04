import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
# from dash import Dash
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

st.title('Stock Price Prediction')

stock_name = "AAPL"
# stock = yf.Ticker(stock_name)

# st.info(stock.major_holders)
# st.info(stock.institutional_holders)
# st.info(stock.mutualfund_holders)
# st.info(stock.news)

start_date = st.date_input("Enter a start date",datetime.date(2020, 1, 1))
st.write(start_date)

end_date = st.date_input("Enter a start date",datetime.date(2022, 1, 1))
st.write(end_date)

#downloaad stock data as per above mentioned dates
stock_data = yf.download(stock_name, start=start_date, end=end_date)
stock_data.to_csv(stock_name)
stock_data = pd.read_csv(stock_name)
#show stock_data
st.info(stock_data.columns)
st.dataframe(stock_data)

fig = go.Figure(

    layout=go.Layout(
        title = f"{stock_name} share price chart"
    ),

    data=[go.Candlestick(
        x=stock_data["Date"],
        open=stock_data["Open"],
        high=stock_data["High"],
        low=stock_data["Low"],
        close=stock_data["Close"],
        increasing_line_color= 'cyan', 
        decreasing_line_color= 'gray'
    )]
)

st.plotly_chart(fig,use_container_width=True)
