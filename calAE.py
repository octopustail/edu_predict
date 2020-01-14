# 计算真实熵
# 统计刷卡次数 -> analyisAE

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
with open('data/record_sequence/record_sequence_10.json', 'r') as f:
    loaded_obj = json.load(f)


# 计算真实熵
def contains(small, big):
    for i in range(len(big)-len(small)+1):
        if big[i:i+len(small)] == small:
            return True
    return False
def actual_entropy(l):
    n = len(l)
    sequence = [l[0]]
    sum_gamma = 0

    for i in range(1, n):
        for j in range(i+1, n+1):
            s = l[i:j]
            if contains(s, sequence) != True:
                sum_gamma += len(s)
                sequence.append(l[i])
                break
    print("n", n, "sum_gamma",sum_gamma)
    ae = (1 / (sum_gamma / n )) * math.log(n)
    return ae
#
# 计算可预测性
def getPredictability(N, S):
    f = lambda x: (((1-x)/(N-1)) **(1-x))* x**x - 2**(-S)
    root = mpmath.findroot(f, 1)
    return float(root.real)


def timepicece(x):
    t = x.split(" ")[1]
    item = datetime.datetime.strptime(t,"%H:%M:%S")
    time_piece = math.floor((item-STAN).seconds/TIME_PIECE)
    return time_piece

ma = lambda x: x.split(" ")

def cal_timepiece(l):
    if len(l)<3:
        return
    nl = np.array(l)
    sp = np.array([timepicece(xi) for xi in nl]).tolist()
    ae = actual_entropy(sp)
    print(ae)
    return ae

# 读取数据

sems = ["sems1","sems2","sems3","sems4","sems5","sems6"]
food_df = pd.DataFrame(columns=sems)
shower_df = pd.DataFrame(columns= sems)
hotwater_df = pd.DataFrame(columns= sems)
library_df = pd.DataFrame(columns= sems)

print("装载DF")
for i, k in loaded_obj.items():
    f = k["food"]
    s = k["shower"]
    h = k["hotwater"]
    l = k["library"]
    food_df.loc[i] = [f["sems1"],f["sems2"],f["sems3"],f["sems4"],f["sems5"],f["sems6"]]
    shower_df.loc[i] = [s["sems1"],s["sems2"],s["sems3"],s["sems4"],s["sems5"],s["sems6"]]
    hotwater_df.loc[i] = [h["sems1"],h["sems2"],h["sems3"],h["sems4"],h["sems5"],h["sems6"]]
    library_df.loc[i] = [l["sems1"],l["sems2"],l["sems3"],l["sems4"],l["sems5"],l["sems6"]]


# 刷卡次数统计
food_df_count  = food_df[sems].applymap(lambda x: len(x))
shower_df_count  = shower_df[sems].applymap(lambda x: len(x))
hotwater_df_count  = hotwater_df[sems].applymap(lambda x: len(x))
library_df_count  = library_df[sems].applymap(lambda x: len(x))

# mapapply 计算每个学期的真实熵
food_df_ae = food_df[sems].applymap(cal_timepiece)
shower_df_ae  = shower_df[sems].applymap(cal_timepiece)
hotwater_df_ae  = hotwater_df[sems].applymap(cal_timepiece)
library_df_ae  = library_df[sems].applymap(cal_timepiece)

food_df_count = food_df_count.join(food_df_ae,rsuffix="ae",lsuffix="count",how="left")
shower_df_count = shower_df_count.join(shower_df_ae,rsuffix="ae",lsuffix="count",how="left")
hotwater_df_count = hotwater_df_count.join(hotwater_df_ae,rsuffix="ae",lsuffix="count",how="left")
library_df_count = library_df_count.join(library_df_ae,rsuffix="ae",lsuffix="count",how="left")

# print("food_count_ae\n",food_count_ae)

filename = ['food','shower','hotwater','library']
dfs = [food_df_count,shower_df_count, hotwater_df_count, library_df_count]
grade = "10"

for i in range(4):
    file = './data/count_ae_'+filename[i]+ '_'+grade+'.pkl'
    df = dfs[i]
    print('start2....')
    print(df)
    df.to_pickle(file)
    print('finish2....')

