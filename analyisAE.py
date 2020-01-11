# 刷卡次数 - 人数画个图


import pandas as pd
import matplotlib.pyplot as plt

filename = ['food','shower','hotwater','library']
grade = "09"

for i in filename:
        file = './data/new_'+i+'_count_'+ grade+'.pkl'
        df = pd.read_pickle(file)
        # df = pd.read_pickle('./data/new_shower_count_10.pkl')
        # df = pd.read_pickle('./data/new_hotwater_count_10.pkl')
        # df = pd.read_pickle('./data/new_library_count_10.pkl')

        print("装载DF")

        df = df.reset_index()


        # print("df",df)
        # s = df["index"].groupby(food_df["sems1"]).count()
        Ys = []
        for i,row in df.iteritems():
            if i == "index":
                continue
            s = df["index"].groupby(df[i]).count()
            s = s[(s>1) | (s<200)].drop(0)
            Ys.append(s)
        print(Ys[4])
        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 1) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[0]) # 画图

        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 2) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[1]) # 画图

        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 3) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[2]) # 画图

        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 4) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[3]) # 画图

        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 5) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[4]) # 画图

        plt.figure(1) # 生成第一个图，且当前要处理的图为fig.1
        plt.subplot(2, 3, 6) # fig.1是一个一行两列布局的图，且现在画的是左图
        plt.plot(Ys[5]) # 画图



        plt.show()


# # 画个刷卡次数统计图




# print('start4....')
# food_df.to_pickle('data/new_food_count_10.pkl')
# print('finish4....')
