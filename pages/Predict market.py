import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import os  
import LSTM_model

st.title('Predict share prices')
#ask user for share name
stock_name = st.text_input('Enter Code-name of your desired share')

c1,c2 = st.columns(2)

with c1:
    start_date = st.date_input("Enter a start date",datetime.date(2016, 1, 1))
    st.write(start_date)
with c2:
    end_date = st.date_input("Enter a start date",datetime.date(2020, 1, 1))
    st.write(end_date)


if len(stock_name) != 0:
    st.info(f'Curernt processing Share "{stock_name}"')
    #downloaad stock data as per above mentioned dates
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    
    os.makedirs('Data', exist_ok=True)  
    stock_data.to_csv(f'Data/{stock_name}.csv')
    
    df = pd.read_csv(f'Data/{stock_name}.csv')
    st.dataframe(df,use_container_width=True)

    LSTM_model.forecast(df)


    