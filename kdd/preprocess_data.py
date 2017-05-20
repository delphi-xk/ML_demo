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
                             ['intersection_id', 'tollgate_id', 'starting_time', 'travel_time']]\
                             .sort_values('starting_time')
    return separate_data

B_1 = select_data('B', 1, trace_data)

# change data to time index
B_1_time_index = DataFrame(B_1['travel_time'].values,
                           index=pd.to_datetime(B_1['starting_time'], format='%Y-%m-%d %H:%M:%S'),
                           columns=['travel_time'])

# resample mean travel time (seconds) in 20Min time intervals, left closed
B_1_resample = B_1_time_index.resample('20Min', closed='left').mean()

# FIX ME: nearly half of the data is null, need some processing or interpolation
# print(B_1_resample.info())

slc = B_1_resample.index.to_series().dt.hour.isin(range(6, 19))
select_time = B_1_resample.loc[slc]
print(select_time)
