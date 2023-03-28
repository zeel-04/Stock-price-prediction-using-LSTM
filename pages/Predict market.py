import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import os  
import LSTM_model

st.set_page_config(layout="wide")

st.title('Predict share prices.')
st.markdown("""---""")

#ask user for share name
stock_list = pd.read_csv("./stock_list.csv")
stock_name = st.selectbox('Enter Code-name of your desired share',stock_list)

c1,c2 = st.columns(2)

with c1:
    start_date = st.date_input("Enter a start date",datetime.date(2016, 1, 1))
    st.write(start_date)
with c2:
    end_date = st.date_input("Enter a end date",datetime.date(2020, 1, 1))
    st.write(end_date)


if len(stock_name) != 0:

    with st.spinner("Loading..."):
        st.info(f'Curernt processing Share **:blue[{stock_name}]**')
        #downloaad stock data as per above mentioned dates
        stock_data = yf.download(stock_name, start=start_date, end=end_date)
    
        os.makedirs('Data', exist_ok=True)  
        stock_data.to_csv(f'Data/{stock_name}.csv')
        
        df = pd.read_csv(f'Data/{stock_name}.csv')

        st.subheader(f"**:blue[{stock_name}]** Stock Data.")
        st.dataframe(df,use_container_width=True)

        with st.spinner("AI is at work..."):
            LSTM_model.forecast(df)


    