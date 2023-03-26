import streamlit as st
import pandas as pd

stock_list = pd.read_csv("./stock_list.csv")

st.selectbox(":blue[Find stock codes]", stock_list)

st.subheader(":blue[Explore all stock codes]")
st.dataframe(stock_list,use_container_width=True)