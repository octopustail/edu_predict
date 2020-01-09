# coding: utf-8
# pylint: disable = invalid-name, C0111
#~/opt/anaconda3/bin/conda install pandas

from __future__ import division

import itertools
import json
import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# load or create your dataset
print('Load data...')
# # 刷卡记录月统计
# food_count_df = pd.read_pickle("data/food_count_df.pkl")
hotwater_count_df = pd.read_pickle("data/hotwater_count_df.pkl")
library_count_df = pd.read_pickle("data/library_count_df.pkl")
# shower_count_df = pd.read_pickle("data/shower_count_df.pkl")
# # 数学成绩
# math_df = pd.read_pickle("data/math_df.pkl")
# # 真实熵
shower_ae_df = pd.read_pickle("data/shower_ae_df.pkl")
meal_ae_df = pd.read_pickle("data/meal_ae_df.pkl")
# 总成绩
total_df = pd.read_pickle("data/total_df.pkl")
total_df['label'] = total_df['2011-2012_2'].map(lambda x:( x>= 60 and False ) or (x<60 and True))
# print(total_df)

# total_df的学号类型从str -> int
# reset_index 的方式用不了，所以采用增加一行的方式
idx = []
for i in total_df.index.to_list():
    idx.append(int(i))
total_df['idx'] = idx

# 数据预处理
# label： 成绩布尔化，及格/不及格

# feature：
# 每月刷卡次数
sems1 = ["2009-09","2009-10","2009-11","2009-12","2010-01","2010-02"]
# food_count_feature_df = food_count_df[sems1]
hotwater_count_feature_df = hotwater_count_df[sems1]
library_count_feature_df = library_count_df[sems1]
# shower_count_feature_df = shower_count_df[sems1]



# feature-label_df
# 连接
print("连接 总成绩 - food刷卡记录...")
res = total_df.join(library_count_feature_df, on="idx", how="outer",sort=True)
res = res.join(hotwater_count_feature_df,on="idx", how="left", sort=True,rsuffix="_hw")
res = res.join(shower_ae_df,on="idx", how="left", sort=True,rsuffix="_shwr")
res = res.join(meal_ae_df,on="idx", how="left", sort=True,rsuffix="_ml")
# 清理掉 NAN 数据
res = res.dropna(axis=0,how='any')
print("组装完成...")
# 真实熵
# 数学成绩
# 前一学期的成绩

# 打印res的所有columns名称
# print(res.columns.to_list())
# 总成绩 = ['2009-2010_1', '2009-2010_2', '2010-2011_1', '2010-2011_2', '2011-2012_1', '2011-2012_2']
score = ['2009-2010_1', '2009-2010_2', '2010-2011_1', '2010-2011_2', '2011-2012_1', '2011-2012_2']
# label = "label"
label = ["label"]
# 图书馆刷卡次数按月统计 = ['2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02']
lib_count = ['2009-09', '2009-10', '2009-11', '2009-12', '2010-01', '2010-02']
# 教学楼打热水次数按月统计 = ['2009-09_hw', '2009-10_hw', '2009-11_hw', '2009-12_hw', '2010-01_hw', '2010-02_hw']
hotwater_count= ['2009-09_hw', '2009-10_hw', '2009-11_hw', '2009-12_hw', '2010-01_hw', '2010-02_hw']
# 洗澡的真实熵 = ['1', '2', '3', '4', '5', '6']
shower_ae = ['1', '2', '3', '4', '5', '6']
# 吃饭的真实熵 = ['1_ml', '2_ml', '3_ml', '4_ml', '5_ml', '6_ml']
meal_ae = ['1_ml', '2_ml', '3_ml', '4_ml', '5_ml', '6_ml']

feature = lib_count + hotwater_count + [shower_ae[0], meal_ae[0]]
print("装载 feature label...")
label = ["label"]
train  = res[:2500]
test = res[2501:]
y_train = train['label']  # training label
y_test = test['label']  # testing label
X_train = train.loc[:, feature]   # training dataset
X_test = test.loc[:, feature]  # testing dataset
#
num = 0
for i in y_test.to_list():
    if i == True:
        num = num+1
print(num)
print(y_test)
#
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 63


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train,pred_leaf=True)

# feature transformation and write result
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_training_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

# predict and get data on leaves, testing data
y_pred = gbm.predict(X_test,pred_leaf=True)

# feature transformation and write result
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

#for i in range(0,len(y_pred)):
#	for j in range(0,len(y_pred[i])):
#		transformed_testing_matrix[i][j * num_leaf + y_pred[i][j]-1] = 1

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))


# Logestic Regression Start
print("Logestic Regression Start")

# load or create your dataset
print('Load data...')

# c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
# for t in range(0,len(c)):
lm = LogisticRegression(penalty='l2', C=0.001) # logestic model construction
lm.fit(transformed_training_matrix, np.ravel(y_train))  # fitting the data

# y_pred_label = lm.predict(transformed_training_matrix )  # For training data
y_pred_label = lm.predict(transformed_testing_matrix)    # For testing data
# y_pred_est = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
y_pred_est = lm.predict_proba(transformed_testing_matrix)
# Give the probabilty on each label

#print('number of testing data is ' + str(len(y_pred_label)))
#print(y_pred_est)

# 评估方式1：
# calculate predict accuracy
#     num = 0
#     for i in range(0,len(y_pred_label)):
#         if y_test[i] == y_pred_label[i]:
#             if y_train[i] == y_pred_label[i]:
#                 num += 1
#     print("*******************")
#     print(y_test)
#     print(y_pred_label)
#     print('penalty parameter is '+ str(c[t]))
#     print("prediction accuracy is " + str((num)/len(y_pred_label)))
#
#     df_prob = pd.DataFrame(
#         y_pred_est,
#         columns=['Death', 'Survived'])
#     fpr, tpr, thresholds = roc_curve(
#         y_test, df_prob['Survived'])
#     # find the area under the curve (auc) for the
#     # ROC
#     roc_auc = auc(fpr, tpr)
#     plt.title(
#         'Receiver Operating Characteristic Curve')
#     plt.plot(fpr, tpr, 'black',
#              label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([-0.1, 1.1])
#     plt.ylim([-0.1, 1.1])
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.show()
#
# 	# Calculate the Normalized Cross-Entropy
# 	# for testing data
#
#     NE = (-1) / len(y_pred_est) * sum(((1+y_test)/2 * np.log(y_pred_est[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_est[:,1])))
# 	# for training data
#     # NE = (-1) / len(y_pred_est) * sum(((1+y_train)/2 * np.log(y_pred_est[:,1]) +  (1-y_train)/2 * np.log(1 - y_pred_est[:,1])))
#     print("Normalized Cross Entropy " + str(NE))

# 评估方式2：

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(15, 10))
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_est[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1

    cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print('Recall metric in the testing dataset while threshold=%s:' % i,
          cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # 画出混淆矩阵
    class_names = [0, 1]

    # plot_confusion_matrix是一个自定义的绘制混淆矩阵图表的函数
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)
plt.show()

