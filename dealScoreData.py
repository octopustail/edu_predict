# 处理学生成绩数据、真实熵数据
# 把学生成绩、数学成绩、真实熵分别导入到dataframe中

import pandas as pd
import numpy as np
import json
import pickle

#读取文件
print("读取文件ing")
# total = json.load(open('/Volumes/Mac/eduData_octo/score/Score_with_courseType/score_wa_total_10.json','r'))

# 导入总成绩
print("导入总成绩ing")
total_df = pd.read_json('/Volumes/Mac/eduData_octo/score/Score_with_courseType/score_wa_total_10.json', orient="records")
print(total_df)

# total_df = pd.DataFrame.from_dict(total, orient="index")
# 只保留六个学期的数据
print("清理无效的学期ing")
# total_df= total_df.drop(['2013-2014_1','2013-2014_2'],axis=1)
# 去掉所有含有NAN的行
print("清理无效的成绩ing")
total_df = total_df.dropna(axis=0,how='any')

print(" ********************* ")
print(total_df)

# # 导入数学成绩
# print("导入数学成绩ing")
# math_list = json.load(open("/Volumes/Mac/eduData_octo/score/mathGrade.json","r"))
#
# # 初始化DF
# print("初始化df")
# idx = ["cal1_m","cal1_f","linear_m","linear_f","cal2_m","cal2_f","prob_m","prob_f"]
# math_df = pd.DataFrame(columns = idx)

# 遍历math数组，填入DF
# print("遍历math数组，填入DF")
# for v in math_list:
#     obj ={"cal1_m":v["cal1"]["midium"],"cal1_f":v["cal1"]["final"],"linear_m":v["linear"]["midium"],"linear_f":v["linear"]["final"],"cal2_m":v["cal2"]["midium"],"cal2_f":v["cal2"]["final"],"prob_m":v["pro"]["midium"],"prob_f":v["pro"]["final"]}    # math_df.loc[v.sid] = obj
#     math_df.loc[v["sid"]] = obj
#
# print(" ********************* ")


# 导入真实熵
print("导入meal_ae")
meal_ae_df = pd.read_csv("/Volumes/Mac/eduData_octo/record_ae_lib/data/mealtime_ae_10.csv",header=None, sep=",").set_index(0)
print("导入shower_ae")
shower_ae_df = pd.read_csv("/Volumes/Mac/eduData_octo/record_ae_lib/data/shower_ae_10.csv",header=None, sep=",").set_index(0)

print(" ********************* ")

# 数据写入pickle

print('start1....')
total_df.to_pickle('data/total_df_10.pkl')
print('finish1....')

print('start2....')
meal_ae_df.to_pickle('data/meal_ae_df_10.pkl')
print('finish2....')

print('start3....')
shower_ae_df.to_pickle('data/shower_ae_df_10.pkl')
print('finish3....')
#
# print('start4....')
# math_df.to_pickle('data/math_df_10.pkl')
# print('finish4....')
