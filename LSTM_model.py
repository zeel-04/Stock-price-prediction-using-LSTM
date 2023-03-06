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

    # st.info(model.summary)

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    st.info(math.sqrt(mean_squared_error(y_train,train_predict)))
    st.info(math.sqrt(mean_squared_error(ytest,test_predict)))

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

    fig = go.Figure()
    trace1 = px.line(scaler.inverse_transform(df_1))
    trace2 = px.line(trainPredictPlot)
    trace2.update_traces(line_color='orange')
    trace3 = px.line(testPredictPlot)
    trace3.update_traces(line_color='green')
    fig.add_trace(trace1.data[0])
    fig.add_trace(trace2.data[0])
    fig.add_trace(trace3.data[0])
    st.plotly_chart(fig,use_container_width=True,theme="streamlit")

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
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
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

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    # fig2 = go.Figure()
    # trace1 = px.line(y=scaler.inverse_transform(df_1[len(df_1)-100:]),x=day_new)
    # trace2 = px.line(y=scaler.inverse_transform(lst_output),x=day_pred)
    # trace2.update_traces(line_color='orange')
    # fig2.add_trace(trace1.data[0])
    # fig2.add_trace(trace2.data[0])
    # st.plotly_chart(fig2,use_container_width=True,theme="streamlit")

    fig = plt.figure()
    plt.plot(day_new,scaler.inverse_transform(df_1[len(df_1)-100:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    st.pyplot(fig)