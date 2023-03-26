import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from collections import deque

def forecast(df):

    df_1=df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0,1))
    arr = np.array(df_1).reshape(-1,1)
    df_1 = scaler.fit_transform(arr)

    #train test splitting
    training_size = int(len(df_1)*0.70)
    test_size = len(df_1) - training_size
    train_data,test_data = df_1[0:training_size,:],df_1[training_size:len(df_1),:1]

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=1,batch_size=64,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    st.subheader("Mean Squared :red[Error].")
    st.markdown("#")
    col1, col2 = st.columns(2)
    col1.metric("Train Prediction",f"{round(math.sqrt(mean_squared_error(y_train,train_predict)))}","- Error")
    col2.metric("Test Prediction",f"{round(math.sqrt(mean_squared_error(ytest,test_predict)))}","- Error")
    st.markdown("#")
    # st.info(math.sqrt(mean_squared_error(y_train,train_predict)))
    # st.info(math.sqrt(mean_squared_error(ytest,test_predict)))

    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = np.empty_like(df_1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df_1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_1)-1, :] = test_predict

    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(df_1))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()

    # fig = go.Figure()
    # trace1 = px.line(scaler.inverse_transform(df_1))
    # trace2 = px.line(trainPredictPlot)
    # trace2.update_traces(line_color='orange')
    # trace3 = px.line(testPredictPlot)
    # trace3.update_traces(line_color='green')
    # fig.add_trace(trace1.data[0])
    # fig.add_trace(trace2.data[0])
    # fig.add_trace(trace3.data[0])
    # fig.update_layout(showlegend=False)
    # st.plotly_chart(fig,use_container_width=True,theme="streamlit")

    st.subheader("Visualizing :blue[Train] and :blue[Test] Prediction.")

    df['Close Price'] = scaler.inverse_transform(df_1)
    df['Train Predict Plot'] = trainPredictPlot
    df['Test Predict Plot'] = testPredictPlot

    figt = px.line(df,x=df['Date'],y=df['Close Price'])
    figt.add_scatter(x=df['Date'],y=df['Train Predict Plot'],name="Train prediction")
    figt.add_scatter(x=df['Date'],y=df['Test Predict Plot'],name="Test prediction",marker=dict(color='orange'))
    st.plotly_chart(figt,use_container_width=True,theme="streamlit")

    x_input = df_1[len(df_1)-100:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 30 days
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    #forecast plotting with matplotlib
    # day_new=np.arange(1,101)
    # day_pred=np.arange(101,131)

    # fig = plt.figure()
    # plt.plot(day_new,scaler.inverse_transform(df_1[len(df_1)-100:]))
    # plt.plot(day_pred,scaler.inverse_transform(lst_output))
    # st.pyplot(fig)

    st.subheader("Forecasting share prices over the next :orange[30 days] using data from the preceding :blue[100 days].")
    #new forecast plotting with plotly
    data = []
    for value in scaler.inverse_transform(df_1[len(df_1)-100:]):
        data.append(value[0])
    
    for value in scaler.inverse_transform(lst_output):
        data.append(value[0])

    end_date = datetime.strptime(max(df['Date']),"%Y-%m-%d")
    dates = []
    for i in range(100):
        dates = deque(dates)
        dates.appendleft(end_date.date() - timedelta(days=i))
        dates = list(dates)

    for i in range(30):
        dates.append(end_date.date() + timedelta(days=(i+1)))

    forecast_df = pd.DataFrame(data, columns=['Value'])
    forecast_df['Date'] = dates

    forecast_plot_fig = px.line(forecast_df, x=forecast_df['Date'][:100],y=forecast_df['Value'][:100])
    forecast_plot_fig.add_scatter(x=forecast_df['Date'][100:],y=forecast_df['Value'][100:],name = 'Forecasted Share Price',marker=dict(color="orange"))
    forecast_plot_fig.update_layout(xaxis_title="Date",yaxis_title="Price")
    st.plotly_chart(forecast_plot_fig,use_container_width=True,theme="streamlit")

    # fig2 = go.Figure()
    # trace1 = px.line(y=scaler.inverse_transform(df_1[len(df_1)-100:]),x=day_new)
    # trace2 = px.line(y=scaler.inverse_transform(lst_output),x=day_pred)
    # trace2.update_traces(line_color='orange')
    # fig2.add_trace(trace1.data[0])
    # fig2.add_trace(trace2.data[0])
    # st.plotly_chart(fig2,use_container_width=True,theme="streamlit")