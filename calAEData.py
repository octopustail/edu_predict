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

# 校历时间转化成datetime格式，方便后续将记录划归到学期
l = ['2009-08-31','2010-03-01','2010-08-31','2011-02-21','2011-08-29','2012-02-20','2012-08-31']
# schoolCalender = map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').date(), l)
li = []
for i in l:
    li.append(datetime.datetime.strptime(i,'%Y-%m-%d'))

def find_semester(date):
    r = range(0, len(li)-1)
    res = 7
    for i in r:
        if (date >= li[i]) & (date < li[i+1]):
            res = i+1
            break
        if(date >= li[6]) | (date<li[1]):
            break
    return res
# 计算真实熵

print("++++++++++++++ Loading Data ++++++++++++++++")
data = pd.read_csv("/Volumes/Mac/eduData_octo/record_ae_lib/data/comsumption_10.csv", header=None, sep=',')

print("-------------- Dropping Data --------------")
# column：0   1    2     3
# 学号  刷卡日期   刷卡时间  刷卡地点
data = data.head(5000)
record_df = data.drop([4, 5, 6], axis=1)
record_df.columns = ["sid","s_date","s_time","s_location"]

# 学生的行为序列（按校历学期划分）
stu_sequence_dict = {}

print("++++++++++++++ Filling stu_sequence_dict ++++++++++++++++")
length = record_df.shape[0]
limit = range(1, 10000)
for i, r in record_df.iterrows():
# for r in limit:
    # 如果stu_sequence_dict中还没有添加该条学生的信息，则在stu_sequence_dict中为该学生初始化
    if r["sid"] not in stu_sequence_dict.keys():
        {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[]}
        stu_sequence_dict[r["sid"]] = {}
        stu_sequence_dict[r["sid"]]["food"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[]}
        stu_sequence_dict[r["sid"]]["shower"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[]}
        stu_sequence_dict[r["sid"]]["hotwater"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[]}
        stu_sequence_dict[r["sid"]]["library"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[]}
    d = datetime.datetime.strptime(r["s_date"]+' '+r["s_time"], '%Y-%m-%d %H:%M:%S')
    sem = find_semester(d)
    print(sem)
    if sem >6:
        continue
    else:
        stu_sequence_dict[r["sid"]][r["s_location"]]["sems"+str(sem)].append(d)
        print(stu_sequence_dict[r["sid"]][r["s_location"]]["sems"+str(sem)])
    print(i, "/" , length)

print(stu_sequence_dict)


#
# 将数据保存在pkl中
print('start1....')
with open('data/sequence.pkl_test', 'wb') as f:
    pickle.dump(stu_sequence_dict, f)
print('finish1....')

# print('start2....')
# print("library_count_df\n",library_count_df)
# library_count_df.to_pickle('data/library_count_df_10.pkl')
# print('finish2....')
#
# print('start3....')
# print("shower_count_df\n",shower_count_df)
# shower_count_df.to_pickle('data/shower_count_df_10.pkl')
# print('finish3....')
#
#
# print('start4....')
# print("hotwater_count_df\n",hotwater_count_df)
# hotwater_count_df.to_pickle('data/hotwater_count_df_10.pkl')
# print('finish4....')
#
#

# def contains(small, big):
#     for i in range(len(big)-len(small)+1):
#         if big[i:i+len(small)] == small:
#             return True
#     return False
# def actual_entropy(l):
#     n = len(l)
#     sequence = [l[0]]
#     sum_gamma = 0
#
#     for i in range(1, n):
#         for j in range(i+1, n+1):
#             s = l[i:j]
#             if contains(s, sequence) != True:
#                 sum_gamma += len(s)
#                 sequence.append(l[i])
#                 break
#
#     ae = 1 / (sum_gamma / n ) * math.log(n)
#     return ae
#
# def getPredictability(N, S):
#     f = lambda x: (((1-x)/(N-1)) **(1-x))* x**x - 2**(-S)
#     root = mpmath.findroot(f, 1)
#     return float(root.real)
# # getPredictability(N, S)
