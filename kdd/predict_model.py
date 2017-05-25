# _*_ coding: utf-8 _*_

"""

build_model created by xiangkun on 2017/5/24

"""
import numpy as np
from keras.models import load_model
import pandas as pd
from pandas import DataFrame
from kdd.preprocess_data import select_data
from kdd.build_model_2 import normalise_window,de_normalise_window
# test data from 10.18 to 10.24, 7 days
# time 6,7,15,16, 4 hours, 12 intervals
# 7*12 = 84 intervals totally




def preproccess_test_data(raw_data):
    n_time_index = DataFrame(raw_data['travel_time'].values,
                             index=pd.to_datetime(raw_data['starting_time'], format='%Y-%m-%d %H:%M:%S'),
                             columns=['travel_time'])
    n_resampled = n_time_index.resample('20Min', closed='left').mean()
    select_time = n_resampled.index.to_series().dt.hour.isin([6,7,15,16])
    n_select_time = n_resampled.loc[select_time]
    n_interpolate = n_select_time.interpolate('linear')
    return n_interpolate


def predict_data(test_array,model):
    predict_array = []
    test_array = np.reshape(test_array,(14,6,1))
    for i in range(14):
        trainX = test_array[i].flatten().tolist()
        for j in range(6):
            # assert len(trainX) == 6
            trainX_nor = normalise_window(trainX)
            pred_y = model.predict(np.reshape(trainX_nor,(1,6,1)))[0][0]
            pred_y_denor = de_normalise_window(trainX,pred_y)
            del trainX[0]
            trainX.append(pred_y_denor)
            predict_array.append(pred_y_denor)
    return predict_array

if __name__ == "__main__":
    a_2_model = load_model('../models/a_2_models.h5')
    # a_x_model = load_model('../models/a_x_models.h5')
    # b_x_model = load_model('../models/b_x_models.h5')
    # c_x_model = load_model('../models/c_x_models.h5')
    test_data = pd.read_csv("../datasets/trajectories(table 5)_test1.csv")

    A_2_test = select_data('A', 2, test_data)
    A_2_processed = preproccess_test_data(A_2_test)
    # A_3_test = select_data('A', 3, test_data)
    # A_3_processed = preproccess_test_data(A_3_test)
    #
    # B_1_test = select_data('B', 1, test_data)
    # B_1_processed = preproccess_test_data(B_1_test)
    # B_3_test = select_data('B', 3, test_data)
    # B_3_processed = preproccess_test_data(B_3_test)
    #
    # C_1_test = select_data('C', 1, test_data)
    # C_1_processed = preproccess_test_data(C_1_test)
    # C_3_test = select_data('C', 3, test_data)
    # C_3_processed = preproccess_test_data(C_3_test)

    a_2_pred = predict_data(A_2_processed.values, a_2_model)
    # a_3_pred = predict_data(A_3_processed.values, c_x_model)
    #
    # b_1_pred = predict_data(B_1_processed.values, c_x_model)
    # b_3_pred = predict_data(B_3_processed.values, c_x_model)
    #
    # c_1_pred = predict_data(C_1_processed.values, c_x_model)
    # c_3_pred = predict_data(C_3_processed.values, c_x_model)

    result = DataFrame({
        "a_2_pred": a_2_pred,
        # "a_3_pred": a_3_pred,
        # "b_1_pred": b_1_pred,
        # "b_3_pred": b_3_pred,
        # "c_1_pred": c_1_pred,
        # "c_3_pred": c_3_pred,
    })

    result.to_csv('../datasets/a_2_pred.csv')