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

# test data from 10.25 to 10.31, 7 days
# time 6,7,15,16, 4 hours, 12 intervals
# 7*12 = 84 intervals totally
# predict time 8,9,17,18


def add_first(df):
    df.loc[pd.to_datetime('2016-10-25 06:00:00', format='%Y-%m-%d %H:%M:%S')] = df.iloc[0]
    df.sort_index()
    return df

def add_last(df):
    df.loc[pd.to_datetime('2016-10-31 16:40:00', format='%Y-%m-%d %H:%M:%S')] = df.iloc[df.size-1]
    df.sort_index()
    return df

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


def generate_time_window(time_series):
    time_array = []
    for t in time_series:
        start = (t+ pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        end = (t +pd.Timedelta(hours=2)+pd.Timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
        time_string = '[{0},{1})'.format(start, end)
        time_array.append(time_string)
    return time_array


def generate_submit_frame(in_id, t_id, window, pred):
    submit = DataFrame({
        'intersection_id': in_id,
        'tollgate_id': t_id,
        'time_window': window,
        'avg_travel_time': pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])
    return submit

if __name__ == "__main__":
    # a_2_model = load_model('../models/a_2_models.h5')
    # a_3_model = load_model('../models/a_3_models.h5')
    a_x_model = load_model('../models/a_x_models.h5')
    b_x_model = load_model('../models/b_x_models.h5')
    c_x_model = load_model('../models/c_x_models.h5')
    test_data = pd.read_csv("../datasets/trajectories(table 5)_test2.csv")

    A_2_test = select_data('A', 2, test_data)
    A_2_processed = preproccess_test_data(A_2_test)
    A_3_test = select_data('A', 3, test_data)
    A_3_processed = preproccess_test_data(A_3_test)

    # B_1 need add last value
    B_1_test = select_data('B', 1, test_data)
    B_1_processed = preproccess_test_data(B_1_test)
    B_1_processed = add_last(B_1_processed)

    B_3_test = select_data('B', 3, test_data)
    B_3_processed = preproccess_test_data(B_3_test)
    #
    C_1_test = select_data('C', 1, test_data)
    C_1_processed = preproccess_test_data(C_1_test)
    # C_3 need add first value
    C_3_test = select_data('C', 3, test_data)
    C_3_processed = preproccess_test_data(C_3_test)
    C_3_processed = add_first(C_3_processed)

    time_window = generate_time_window(A_2_processed.index)

    a_2_pred = predict_data(A_2_processed.values, a_x_model)
    a_3_pred = predict_data(A_3_processed.values, a_x_model)
    #
    b_1_pred = predict_data(B_1_processed.values, b_x_model)
    b_3_pred = predict_data(B_3_processed.values, b_x_model)

    c_1_pred = predict_data(C_1_processed.values, c_x_model)
    c_3_pred = predict_data(C_3_processed.values, c_x_model)

    # result = DataFrame({
    #     "a_2_pred": a_2_pred,
    #     "a_3_pred": a_3_pred,
    #     "b_1_pred": b_1_pred,
    #     "b_3_pred": b_3_pred,
    #     "c_1_pred": c_1_pred,
    #     "c_3_pred": c_3_pred,
    # })
    #
    # result.to_csv('../datasets/result_pred.csv')


    A_2_format = DataFrame({
        'intersection_id': 'A',
        'tollgate_id': '2',
        'time_window': time_window,
        'avg_travel_time': a_2_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    A_3_format = DataFrame({
        'intersection_id': 'A',
        'tollgate_id': '3',
        'time_window': time_window,
        'avg_travel_time': a_3_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    B_1_format = DataFrame({
        'intersection_id': 'B',
        'tollgate_id': '1',
        'time_window': time_window,
        'avg_travel_time': b_1_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    B_3_format = DataFrame({
        'intersection_id': 'B',
        'tollgate_id': '3',
        'time_window': time_window,
        'avg_travel_time': b_3_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    C_1_format = DataFrame({
        'intersection_id': 'C',
        'tollgate_id': '1',
        'time_window': time_window,
        'avg_travel_time': c_1_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    C_3_format = DataFrame({
        'intersection_id': 'C',
        'tollgate_id': '3',
        'time_window': time_window,
        'avg_travel_time': c_3_pred
    }, columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'])

    # assert A_2_format.size == A_3_format.size == B_1_format.size == B_3_format.size == C_1_format.size == C_3_format.size == 84
    print(A_2_format.info())
    print(A_3_format.info())
    print(B_1_format.info())
    print(B_3_format.info())
    print(C_1_format.info())
    print(C_3_format.info())


    result_submit = pd.concat([
        A_2_format, A_3_format,
        B_1_format, B_3_format,
        C_1_format, C_3_format
    ])

    result_submit.to_csv("../datasets/result_submit.csv", index=False)
