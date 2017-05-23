# _*_ coding: utf-8 _*_

"""

build_model created by xiangkun on 2017/5/16

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras import losses
from keras import backend as K
# Training datasets from 07.19 - 10.17, total 91 days
# Consider time from 6 to 19, that is 13 hours a day
# Total datasets length 91*13*3 = 3549
datasets = pd.read_csv('../datasets/A_2_processed.csv')
dataArray = datasets['travel_time'].values.reshape((91, 39))


# 20Min * 6 = 2H
sequence_length = 6


def my_custom_loss(y_true, y_pred):

    pass

def create_dateset(dataArr, sequence_length):
    dataX, dataY = [], []
    for i in range(dataArr.shape[0]):
        for j in range(dataArr.shape[1]-sequence_length-1):
            a = dataArr[i, j:j+sequence_length]
            b = dataArr[i, j+sequence_length]
            dataX.append(a)
            dataY.append(b)
    return np.array(dataX), np.array(dataY)

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0,1))
# dataArray = scaler.fit_transform(dataArray)


trainX, trainY = create_dateset(dataArray, sequence_length)


# reshape input data to be [sample, time step, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))



# create and fit LSTM model
model = Sequential()
model.add(LSTM(
    3, input_shape=(sequence_length, 1), return_sequences=True))
model.add(LSTM(6, input_shape=(sequence_length, 1), return_sequences=True))
model.add(LSTM(13, input_shape=(sequence_length, 1), return_sequences=True))
# model.add(LSTM(30, input_shape=(sequence_length, 1), return_sequences=True))
# model.add(LSTM(30, input_shape=(sequence_length, 1), return_sequences=True))
model.add(LSTM(13, input_shape=(sequence_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(6, input_shape=(sequence_length, 1), return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))

# use metric MAPE
# Equivalent to MAE, but sometimes easier to interpret.
# diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
# return 100. * K.mean(diff, axis=-1)

model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
# model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1, validation_split=0.2)

