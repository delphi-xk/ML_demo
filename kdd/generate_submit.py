# _*_ coding: utf-8 _*_

"""

generate_submit created by xiangkun on 2017/5/31

"""
import pandas as pd


def generate_time_window(time_series):
    time_array = []
    for t in time_series:
        start = t.strftime('%Y-%m-%d %H:%M:%S')
        end = (t + pd.Timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
        time_string = '[{0},{1})'.format(start, end)
        time_array.append(time_string)
    return time_array

if __name__ == '__main__':
    pd.read_csv("../datasets/result_pred.csv")
