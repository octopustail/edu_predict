# 处理学生记录数据
# 学生的刷卡次数按月统计

import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("/Volumes/Mac/eduData_octo/record_ae_lib/data/comsumption_10.csv", header=None, sep=',')

# column：0   1   3
# 学号  刷卡日期   刷卡地点
record_df = data.drop([2, 4, 5, 6], axis=1)

print('按月统计每人每类的刷卡次数')
# 按月统计每人每类的刷卡次数
record_df[1] = record_df[1].map(lambda x: x[:7])
counts = record_df[0].groupby([record_df[0], record_df[1], record_df[3]]).count()

# 学生的IDlist
_total = record_df[0].drop_duplicates()
total = np.array(_total).tolist()

# 初始化feature df
sems = ['2010-09','2010-10','2010-11','2010-12','2011-01','2011-02','2011-03','2011-04','2011-05','2011-06','2011-07','2011-08','2011-09','2011-10','2011-11','2011-12','2012-01','2012-02','2012-03','2012-04','2012-05','2012-06','2012-07','2012-08','2012-09','2012-10','2012-11','2012-12','2013-01','2013-02','2013-03','2013-04','2013-05','2013-06']

print('初始化feature df')
food_count_df = pd.DataFrame(np.zeros((len(total), len(sems))), index=total, columns=sems, dtype=np.int)
shower_count_df = pd.DataFrame(np.zeros((len(total), len(sems))), index=total, columns=sems, dtype=np.int)
library_count_df = pd.DataFrame(np.zeros((len(total), len(sems))), index=total, columns=sems, dtype=np.int)
hotwater_count_df = pd.DataFrame(np.zeros((len(total), len(sems))), index=total, columns=sems, dtype=np.int)

len_c = len(counts)
print('start 填充数据')
for i, v in counts.items():
    # 如果counts里的日期大于sems里的最大日期，则不处理该条记录
    if (i[0] in total) & (i[1] in sems):
        col = total.index(i[0])
        idx = sems.index(i[1])
        print(col, "/", len(total))
        # 将记录分类填写到对应的dataframe中
        if 'food' in i:
            food_count_df.iloc[col, idx] = v
            continue
        if 'shower' in i:
            shower_count_df.iloc[col, idx] = v
            continue


        if 'library' in i:
            library_count_df.iloc[col, idx] = v
            continue
        if 'hotwater' in i:
            hotwater_count_df.iloc[col, idx] = v
    else:
        continue

# 将几个数据保存在pkl中
print('start1....')
print("food_count_df\n",food_count_df)
food_count_df.to_pickle('data/food_count_df_10.pkl')
print('finish1....')

print('start2....')
print("library_count_df\n",library_count_df)
library_count_df.to_pickle('data/library_count_df_10.pkl')
print('finish2....')

print('start3....')
print("shower_count_df\n",shower_count_df)
shower_count_df.to_pickle('data/shower_count_df_10.pkl')
print('finish3....')


print('start4....')
print("hotwater_count_df\n",hotwater_count_df)
hotwater_count_df.to_pickle('data/hotwater_count_df_10.pkl')
print('finish4....')



