# _*_ coding: utf-8 _*_

"""

preprocess_data created by xiangkun on 2017/5/16

"""
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
trajectories
columns  "intersection_id","tollgate_id","vehicle_id","starting_time","travel_seq","travel_time"
only intersection_id, tollgate_id, starting_time, travel_time in use so far  
"""

trace_data = pd.read_csv("../datasets/trajectories(table 5)_training.csv")


# FIX ME if more parameters considered
def select_data(intersection_id, tollgate_id, data):
    separate_data = data.loc[(data['intersection_id'] == intersection_id) & (data['tollgate_id'] == tollgate_id),
                             ['intersection_id', 'tollgate_id', 'starting_time', 'travel_time']] \
        .sort_values('starting_time')
    return separate_data


# change data to time index,
# resample into 20Min,
# get time interval from 6 to 19
# interpolate data with linear method
def preprocess_data(raw_data):
    n_time_index = DataFrame(raw_data['travel_time'].values,
                             index=pd.to_datetime(raw_data['starting_time'], format='%Y-%m-%d %H:%M:%S'),
                             columns=['travel_time'])
    # resample mean travel time (seconds) in 20Min time intervals, left closed
    n_resampled = n_time_index.resample('20Min', closed='left').mean()
    # select time from 6 to 19
    select_time = n_resampled.index.to_series().dt.hour.isin(range(6, 19))
    n_select_time = n_resampled.loc[select_time]
    # FIX ME: nearly half of the data is null, need some processing or interpolation
    # print(B_1_resample.info())
    # temporarily use linear
    n_interpolate = n_select_time.interpolate('linear')
    return n_interpolate

# test
# C_3 = select_data('C', 3, trace_data)
# C_3_processed = preprocess_data(C_3)
# print(C_3_processed.isnull().values.any())


route = {
    "A": (2, 3),
    "B": (1, 3),
    "C": (1, 3)
}


def generate_data_files(dic):
    for i_id in dic.keys():
        for t_id in dic[i_id]:
            x_n = select_data(i_id, t_id, trace_data)
            x_n_processed = preprocess_data(x_n)
            file_name = '{}_{}_processed.csv'.format(i_id, t_id)
            x_n_processed.to_csv('../datasets/{}'.format(file_name))

# generate_data_files(route)

