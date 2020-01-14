# This Python file uses the following encoding: utf-8
'''
@Author: your name
@Date: 2020-01-11 16:24:25
@LastEditTime : 2020-01-12 20:23:18
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /test/calPrediction.py
'''
# 计算真实熵


import pandas as pd
import numpy as np
import math
import mpmath
import pickle
import datetime
import time
import json


# 零点时间
STAN = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
# 时间区间是10分钟
TIME_PIECE = 10 * 60

print("load data...")

#计算可预测性
def getPredictability(N, S):
    if N <= 3 :
        return
    f = lambda x: (((1-x)/(N-1)) **(1-x))* x**x - 2**(-S)
    root = mpmath.findroot(f, 1)
    res = float(root.real)
    return res


# 读取数据
# df = pd.read_pickle('./data/food_count_ae_09.pkl')


filename = ['food','shower','hotwater','library']
# filename = ['food']
grade = "10"

for i in filename:
    file = './data/prediction_ae_countList/count_ae_'+i +'_' + grade+'.pkl'
    df = pd.read_pickle(file)

    df = df.dropna(axis=0, how="any")
    #处理一下缺失值
    ae_df = df[["sems3ae", "sems4ae", "sems5ae","sems6ae"]]
    df = df.fillna(method="bfill",axis=1)
    df = df.dropna(axis=0, how="any")
    # df = df.ix[~(df==0).all(axis=1), :]  # 删了它
    # 2906303015
    for j in range(1,7):
        df["sems"+str(j)+"pred"] = df.apply(lambda row: getPredictability(row['sems'+str(j)+'count'],row['sems'+str(j)+'ae']),axis=1)
    df = df.fillna(method="bfill",axis=1)
    df = df.fillna(method="ffill",axis=1)
    df = df.dropna(axis=0, how="any")

    print(df)
    print('start '+i+ '....')
    print(df)
    outfile = 'data/prediction_ae_countList/prediction_'+ i + grade+'.pkl'
    df.to_pickle(outfile)
    print('finish '+i+'....')

#
# print('start1....')
# print(food_count_ae)
# food_count_ae.to_pickle('data/food_count_ae_09.pkl')
# print('finish1....')
#
# print('start2....')
# print(hotwater_df)
# hotwater_df.to_pickle('data/new_hotwater_count_09.pkl')
# print('finish2....')
# #
# print('start3....')
# print(library_df)
# library_df.to_pickle('data/new_library_count_09.pkl')
# print('finish3....')
# #
# print('start4....')
# print(food_df)
# food_df.to_pickle('data/new_food_count_09.pkl')
# print('finish4....')
