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
#  09年使用
# l = ['2009-08-31','2010-03-01','2010-08-31','2011-02-21','2011-08-29','2012-02-20','2012-08-31']
# 10年使用
l = ['2010-08-31','2011-02-21','2011-08-29','2012-02-20','2012-08-31','2013-02-28','2013-08-31',]
# schoolCalender = map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').date(), l)
# 校历转化为时间戳格式
li = []
for i in l:
    li.append(datetime.datetime.strptime(i,'%Y-%m-%d'))

# 用来判断记录在哪一个学期
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


# #读取数据
# print("++++++++++++++ Loading Data ++++++++++++++++")
data = pd.read_csv("/Users/vis/Desktop/comsumption_10.csv", header=None, sep=',')

print("-------------- Dropping Data --------------")
# column：0   1    2     3
# 学号  刷卡日期   刷卡时间  刷卡地点
record_df = data.drop([4, 5, 6], axis=1)
record_df.columns = ["sid","s_date","s_time","s_location"]

# 学生的行为序列（按校历学期划分）
stu_sequence_dict = {}

print("++++++++++++++ Filling stu_sequence_dict ++++++++++++++++")
length = record_df.shape[0]
for i, r in record_df.iterrows():
    # 如果stu_sequence_dict中还没有添加该条学生的信息，则在stu_sequence_dict中为该学生初始化
    if r["sid"] not in stu_sequence_dict.keys():
        {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[],"sems7":[]}
        stu_sequence_dict[r["sid"]] = {}
        stu_sequence_dict[r["sid"]]["food"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[],"sems7":[]}
        stu_sequence_dict[r["sid"]]["shower"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[],"sems7":[]}
        stu_sequence_dict[r["sid"]]["hotwater"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[],"sems7":[]}
        stu_sequence_dict[r["sid"]]["library"] = {"sems1":[], "sems2":[], "sems3":[], "sems4":[], "sems5":[], "sems6":[],"sems7":[]}
    # 把date和time拼成和校历格式一样，然后去判断是哪一个学期
    d = datetime.datetime.strptime(r["s_date"]+' '+r["s_time"], '%Y-%m-%d %H:%M:%S')
    sem = find_semester(d)
    # 舍去1-6学期以外的数据
    # if sem >6:
    #     continue
    # else:
        # 对号入座到对饮的学期和类型
    stu_sequence_dict[r["sid"]][r["s_location"]]["sems"+str(sem)].append(r["s_date"]+' '+r["s_time"])
    print(i, "/" , length)

print(stu_sequence_dict)


#
# 将数据保存在json中
print('start1....')
json_str = json.dumps(stu_sequence_dict, indent=4)
with open('/Users/vis/Desktop/eduData_octo/record_sequence_3_10.json', 'w') as json_file:
    json_file.write(json_str)
print('finish1....')

