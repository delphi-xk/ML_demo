# _*_ coding: utf-8 _*_

"""

valuate_model created by xiangkun on 2017/5/25

"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from kdd.preprocess_data import select_data



def preprocess_fact_data(raw_data):
    n_time_index = DataFrame(raw_data['travel_time'].values,
                             index=pd.to_datetime(raw_data['starting_time'], format='%Y-%m-%d %H:%M:%S'),
                             columns=['travel_time'])
    # resample mean travel time (seconds) in 20Min time intervals, left closed
    n_resampled = n_time_index.resample('20Min', closed='left').mean()
    # select time from 6 to 19
    select_time = n_resampled.index.to_series().dt.hour.isin([8,9,17,18])
    n_select_time = n_resampled.loc[select_time]
    # FIX ME: nearly half of the data is null, need some processing or interpolation
    # print(B_1_resample.info())
    # temporarily use linear
    n_interpolate = n_select_time.interpolate('linear')
    return n_interpolate

def computeMAPE(y_pred, y_true):
    diff = np.abs((y_true - y_pred) / y_true)
    return diff

if __name__ == '__main__':

    training_2_data = pd.read_csv("../datasets/trajectories(table 5)_training2.csv")
    pred_data = pd.read_csv("../datasets/a_2_pred.csv")
    a_2_fact = preprocess_fact_data(select_data('A',2,training_2_data))['travel_time'].values
    a_2_pred = pred_data['a_2_pred'].values
    assert len(a_2_fact) == len(a_2_pred)
    val = .0
    for i in range(len(a_2_fact)):
        val = val + computeMAPE(a_2_pred[i], a_2_fact[i])
    print(val/len(a_2_fact))

