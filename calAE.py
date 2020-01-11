# 处理学生记录数据
# 学生的刷卡次数按月统计

import pandas as pd
import numpy as np
import math
# import mpmath
import pickle
import datetime
import time
import json
import matplotlib.pyplot as plt


# 零点时间
STAN = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
# 时间区间是10分钟
TIME_PIECE = 10 * 60


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
# # 计算可预测性
# def getPredictability(N, S):
#     f = lambda x: (((1-x)/(N-1)) **(1-x))* x**x - 2**(-S)
#     root = mpmath.findroot(f, 1)
#     return float(root.real)


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
print("load data...")
with open('./data/record_sequence_2_2.json','r') as f:
    loaded_obj = json.load(f)

food_df = pd.DataFrame(columns=["sems1","sems2","sems3","sems4","sems5","sems6"])
shower_df = pd.DataFrame(columns=["sems1","sems2","sems3","sems4","sems5","sems6"])
hotwater_df = pd.DataFrame(columns=["sems1","sems2","sems3","sems4","sems5","sems6"])
library_df = pd.DataFrame(columns=["sems1","sems2","sems3","sems4","sems5","sems6"])

print("装载DF")
for i, k in loaded_obj.items():
    f = k["food"]
    s = k["shower"]
    h = k["hotwater"]
    l = k["library"]
    # print(f["sems1"],f["sems2"],f["sems3"],f["sems4"],f["sems5"],f["sems6"])
    food_df.loc[i] = [f["sems1"],f["sems2"],f["sems3"],f["sems4"],f["sems5"],f["sems6"]]
    shower_df.loc[i] = [s["sems1"],s["sems2"],s["sems3"],s["sems4"],s["sems5"],s["sems6"]]
    hotwater_df.loc[i] = [h["sems1"],h["sems2"],h["sems3"],h["sems4"],h["sems5"],h["sems6"]]
    library_df.loc[i] = [l["sems1"],l["sems2"],l["sems3"],l["sems4"],l["sems5"],l["sems6"]]


# 刷卡次数统计
food_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = food_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(lambda x: len(x))
shower_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = shower_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(lambda x: len(x))
hotwater_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = hotwater_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(lambda x: len(x))
library_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = library_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(lambda x: len(x))

# for i,row in food_df.iteritems():
#     food_df[i] = row.sort_values()
#     print(food_df[i])
# print(food_df)
# # 画个刷卡次数统计图
# food_df[["sems1","sems2"]].plot()
# plt.show()





# mapapply 计算每个学期的真实熵
# food_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = food_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(cal_timepiece)
# shower_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = shower_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(cal_timepiece)
# hotwater_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = hotwater_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(cal_timepiece)
# library_df[["sems1","sems2","sems3","sems4","sems5","sems6"]]  = library_df[["sems1","sems2","sems3","sems4","sems5","sems6"]].applymap(cal_timepiece)


# print('start4....')
# food_df.to_pickle('data/new_food_count_10.pkl')
# print('finish4....')

print('start1....')
print(shower_df)
shower_df.to_pickle('data/new_shower_count_10.pkl')
print('finish1....')

print('start2....')
print(hotwater_df)
hotwater_df.to_pickle('data/new_hotwater_count_10.pkl')
print('finish2....')
#
print('start3....')
print(library_df)
library_df.to_pickle('data/new_library_count_10.pkl')
print('finish3....')
#
print('start4....')
print(food_df)
food_df.to_pickle('data/new_food_count_10.pkl')
print('finish4....')
