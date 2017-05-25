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

A_2_Array = pd.read_csv('../datasets/A_2_processed.csv')['travel_time']\
    .values.reshape((91, 39))

A_3_Array = pd.read_csv('../datasets/A_3_processed.csv')['travel_time']\
    .values.reshape((91, 39))

B_1_Array = pd.read_csv('../datasets/B_1_processed.csv')['travel_time']\
    .values.reshape((91, 39))

B_3_Array = pd.read_csv('../datasets/B_3_processed.csv')['travel_time']\
    .values.reshape((91, 39))

C_1_Array = pd.read_csv('../datasets/C_1_processed.csv')['travel_time']\
    .values.reshape((91, 39))

C_3_Array = pd.read_csv('../datasets/C_3_processed.csv')['travel_time']\
    .values.reshape((91, 39))

# 20Min * 6 = 2H
sequence_length = 6

# use metric MAPE
# Equivalent to MAE, but sometimes easier to interpret.
# diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
# return 100. * K.mean(diff, axis=-1)
def my_custom_loss(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / y_true)
    return diff


def create_dateset(dataArr, sequence_length):
    # MinMaxScaler normalize the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataArr = scaler.fit_transform(dataArr)

    dataX, dataY = [], []
    for i in range(dataArr.shape[0]):
        for j in range(dataArr.shape[1]-sequence_length-1):
            window_data = dataArr[i, j:j+sequence_length+1]
            normalised_data = normalise_window(window_data)
            dataX.append(normalised_data[:sequence_length])
            dataY.append(normalised_data[sequence_length])
    return np.array(dataX), np.array(dataY)

# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0,1))
# dataArray = scaler.fit_transform(dataArray)


# normalize the window data
def normalise_window(window_data):
    normalised_data = [((float(p) / float(window_data[0])) - 1) for p in window_data]
    return normalised_data


def de_normalise_window(window_data,pred):
    return window_data[0]*(1+pred)


# A_2_trainX, A_2_trainY = create_dateset(A_2_Array, sequence_length)
# A_3_trainX, A_3_trainY = create_dateset(A_3_Array, sequence_length)
# print(A_2_trainX.shape, A_2_trainY.shape)
#
# trainX = np.concatenate((A_2_trainX, A_3_trainX), axis=0)
# trainY = np.concatenate((A_2_trainY, A_3_trainY), axis=0)
# print(trainX.shape, trainY.shape)

# generate aggregate datasets
def create_train_data():
    A_2_trainX, A_2_trainY = create_dateset(A_2_Array, sequence_length)
    A_3_trainX, A_3_trainY = create_dateset(A_3_Array, sequence_length)
    # B_1_trainX, B_1_trainY = create_dateset(B_1_Array, sequence_length)
    # B_3_trainX, B_3_trainY = create_dateset(B_3_Array, sequence_length)
    # C_1_trainX, C_1_trainY = create_dateset(C_1_Array, sequence_length)
    # C_3_trainX, C_3_trainY = create_dateset(C_3_Array, sequence_length)
    X = np.concatenate((A_2_trainX, A_3_trainX,
                        # B_1_trainX, B_3_trainX,
                        # C_1_trainX, C_3_trainX
                        ), axis=0)
    Y = np.concatenate((A_2_trainY, A_3_trainY,
                        # B_1_trainY, B_3_trainY,
                        # C_1_trainY, C_3_trainY
                        ), axis=0)
    return X, Y

# trainX, trainY = create_dateset(B_1_Array, sequence_length)
# print(trainX.shape, trainY.shape)
# reshape input data to be [sample, time step, features]

# create and fit LSTM model
def build_model(trainX, trainY):
    model = Sequential()
    model.add(LSTM(
        3, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(LSTM(6,  return_sequences=True))
    model.add(LSTM(13, return_sequences=True))
    model.add(LSTM(13,  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(6, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('linear'))
    # model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1, validation_split=0.1)
    return model

if __name__ == "__main__":
    # trainX, trainY = create_train_data()
    trainX, trainY = create_dateset(A_2_Array, sequence_length)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    # print(trainX.shape, trainY.shape)
    c_x_model = build_model(trainX, trainY)
    c_x_model.save('../models/a_2_models.h5')
    # print(c_x_model.predict(np.reshape([ 0.5  ,  0.7 ,  0.2  , 0.33,  0.45 ,  0.67],(1,6,1))))
    # print(c_x_model.predict(np.reshape([ 0.7 ,  0.2  , 0.33,  0.45 ,  0.67, 0.44 ],(1,6,1))))
