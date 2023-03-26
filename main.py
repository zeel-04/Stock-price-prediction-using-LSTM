import streamlit as st
import yfinance as yf
import datetime
import os  
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

stock_list = pd.read_csv("./stock_list.csv")

st.title('Explore your desired stock.')
# st.markdown("""---""")
st.markdown('#')
st.subheader("Visualize interactive :blue[Open], :blue[Close], :green[High], :red[Low] and :blue[Volume] charts made with :blue[Plotly].")
st.caption("Get information about :blue[Major holders], :blue[Institutional holders] and :blue[Mutual Fund holders].")
st.markdown("""---""")
#ask user for share name
stock_name = st.selectbox('Enter Code-name of your desired share',stock_list)

c1,c2 = st.columns(2)

with c1:
    start_date = st.date_input("Enter a start date",datetime.date(2020, 1, 1))
    st.write(start_date)
with c2:
    end_date = st.date_input("Enter a end date",datetime.date(2022, 1, 1))
    st.write(end_date)

if len(stock_name) != 0:

    with st.spinner("Loading..."):
        st.info(f'Curernt processing Share **:blue[{stock_name}].**')

        #downloaad stock data as per above mentioned dates
        stock_data = yf.download(stock_name, start=start_date, end=end_date)
        os.makedirs('Data', exist_ok=True)
        stock_data.to_csv(f'Data/{stock_name}.csv')
        stock_data = pd.read_csv(f'Data/{stock_name}.csv')

    st.markdown("#")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open Price", f"{round(max(stock_data['Open']))}", "+ Max")
    col1.metric("Open Price", f"{round(min(stock_data['Open']))}", "- Min")
    col2.metric("Close Price", f"{round(max(stock_data['Close']))}", "+ Max")
    col2.metric("Close Price", f"{round(min(stock_data['Close']))}", "- Min")
    col3.metric("High", f"{round(max(stock_data['High']))}", "+ Max")
    col3.metric("High", f"{round(min(stock_data['High']))}", "- Min")
    col4.metric("Low", f"{round(max(stock_data['Low']))}", "+ Max")
    col4.metric("Low", f"{round(min(stock_data['Low']))}", "- Min")
    st.markdown("#")

    #convert stock data to utf-8 format
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(stock_data)

    c1,c2 = st.columns(2)
    with c1:
        #show stock_data
        st.write("To **:blue[display stock data]** click on the show data button.")
        b1 = st.button('Show Data')
    with c2:
        #download stock_data
        st.write("To **:blue[download stock data]** click on the download data button.")
        b2 = st.download_button(label='Download Data',data=csv,file_name=f"{stock_name} data.csv",mime='text/csv',key='download_data')
    
    if b1:
        st.dataframe(stock_data,use_container_width=True)
    if b2:
        st.download_button(label='Download Data',data=csv,file_name=f"{stock_name} data.csv",mime='text/csv')

    st.markdown("#")

    b1,b2 = st.columns(2)
    with b1:
        with st.spinner("Loading..."):
            st.subheader("Open price data visualization.")
            fig2 = go.Figure()
            trace1 = px.line(x=stock_data["Date"],y=stock_data['Open'])
            fig2.add_trace(trace1.data[0])
            fig2.update_layout(xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig2,use_container_width=True,theme="streamlit")
    with b2:
        with st.spinner("Loading..."):
            st.subheader("Close price data visualization.")
            fig2 = go.Figure()
            trace1 = px.line(x=stock_data["Date"],y=stock_data['Close'])
            fig2.add_trace(trace1.data[0])
            fig2.update_layout(xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    c1,c2 = st.columns(2)
    with c1:
        with st.spinner("Loading..."):
            st.subheader("High price data visualization.")
            fig2 = go.Figure()
            trace1 = px.line(x=stock_data["Date"],y=stock_data['High'])
            fig2.add_trace(trace1.data[0])
            fig2.update_layout(xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig2,use_container_width=True,theme="streamlit")
    with c2:
        with st.spinner("Loading..."):
            st.subheader("Low price data visualization.")
            fig2 = go.Figure()
            trace1 = px.line(x=stock_data["Date"],y=stock_data['Low'])
            fig2.add_trace(trace1.data[0])
            fig2.update_layout(xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    with st.spinner("Loading..."):
        st.subheader("Volume data visualization.")
        fig3 = go.Figure()
        trace2 = px.bar(x=stock_data['Date'],y=stock_data['Volume'])
        fig3.add_trace(trace2.data[0])
        fig3.update_layout(xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig3,use_container_width=True,theme="streamlit")


    #################################################################
    st.write("To **:blue[display Candlestick chart with volume]** click on the View button.")
    if st.button('View'):
        with st.spinner("Loading..."):
            # Create subplots and mention plot grid size
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
                        row_width=[0.2, 0.7])

            # Plot OHLC on 1st row
            fig.add_trace(go.Candlestick(x=stock_data["Date"], open=stock_data["Open"], high=stock_data["High"],
                            low=stock_data["Low"], close=stock_data["Close"], name="OHLC"), 
                            row=1, col=1
            )

            # Bar trace for volumes on 2nd row without legend
            fig.add_trace(go.Bar(x=stock_data['Date'], y=stock_data['Volume'], showlegend=False), row=2, col=1)

            # Do not show OHLC's rangeslider plot 
            fig.update(layout_xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig,use_container_width=True,theme="streamlit")
    #################################################################

    st.markdown("#")

    #----------------------------------------------------------------------------
    stock = yf.Ticker(stock_name)

    with st.spinner("Loading..."):
        #major holders
        st.subheader(f"**Major holders of :blue[{stock_name}].**")
        major_holders = stock.major_holders
        major_holders.columns = ['Pecentage','Holders']
        st.dataframe(major_holders,use_container_width=True)

    with st.spinner("Loading..."):
        #institutional_holders
        st.subheader(f"**Institutional holders of :blue[{stock_name}].**")
        institutional_holders = stock.institutional_holders
        st.dataframe(institutional_holders,use_container_width=True)

    with st.spinner("Loading..."):
        #mutualfund_holders
        st.subheader(f"**Mutual Fund holders of :blue[{stock_name}].**")
        institutional_holders = stock.institutional_holders
        st.dataframe(institutional_holders,use_container_width=True)

    with st.spinner("Loading..."):
        st.subheader(f"**News related to :blue[{stock_name}].**")
        stock_news = stock.news
        for news in stock_news:
            st.json(news) 
    # st.info(type(stock.news))
    # st.info(stock.news)
    #----------------------------------------------------------------------------
    # fig = go.Figure(

    #     layout=go.Layout(
    #         title = f"{stock_name} Share price chart"
    #     ),

    #     data=[go.Candlestick(
    #         x=stock_data["Date"],
    #         open=stock_data["Open"],
    #         high=stock_data["High"],
    #         low=stock_data["Low"],
    #         close=stock_data["Close"],
    #         increasing_line_color= 'cyan', 
    #         decreasing_line_color= 'gray',
    #     )]
    # )

    # st.plotly_chart(fig,use_container_width=True,theme="streamlit")

    # fig2 = go.Figure()
    # trace1 = px.line(stock_data,x='Date',y='Close',title='Close trace')
    # trace2 = px.line(stock_data,x='Date',y='Open',title='Open trace')
    # trace2.update_traces(line_color='gray')
    # fig2.add_trace(trace1.data[0])
    # fig2.add_trace(trace2.data[0])
    # st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    

    
