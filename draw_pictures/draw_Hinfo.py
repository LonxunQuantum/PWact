"""
@Description :
读取文件，文件格式 str "[fl,fl,...]"
@Returns     :
@Author       :wuxingxing
"""
from ftplib import error_perm
from mailbox import linesep
import math
from operator import index
import os
from turtle import color
import numpy as np
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
from torch import rand
from draw_pictures.workdir import WorkTrainDir
import seaborn as sns
from scipy.stats import linregress
picture_save_dir = "./mlff_wu_work_dir/picture_dir"

color_list = ["#000000", "#008B8B", "#BDB76B", "#B8860B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+"]
split_list = ["0.05", "0.1", "0.15", "0.2", "uper"]

def read_data():
    df = pd.read_csv(al_pm.valid_loss_H_path)
    indexs = df.index
    for i in indexs:
        if df['HPHt'][i] > 0.3:
            df.drop(i, inplace=True)
    print(df.index.__len__())
    return df

def draw_H_Etot():
    df = read_data()
    fitting_line(plt, df, ['E_tot','HPHt'])
    plt.title("E_tot with HPHt %d images"%df.index.__len__())
    plt.xlabel("E_tot")
    plt.ylabel("HPHt")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(al_pm.valid_dir, "0.3_valid_Etot_H.png"))

def draw_H_loss():
    df = read_data()
    fitting_line(plt, df, ['Loss','HPHt'])
    plt.title("Loss with HPHt %d images"%df.index.__len__())  # 标题
    plt.xlabel("Loss")  # 设置X轴标签
    plt.ylabel("HPHt")  # 设置Y轴标签
    plt.savefig(os.path.join(al_pm.valid_dir, "0.3_valid_Loss_H.png"))

def draw_H_Ef():
    df = read_data()
    fitting_line(plt, df, ['E_f','HPHt'])
    plt.title("E_f with HPHt %d images"%df.index.__len__())  # 标题
    plt.scatter(df['E_f'], df['HPHt']) # 散点图
    plt.xlabel("E_f")  # 设置X轴标签
    plt.ylabel("HPHt")  # 设置Y轴标签
    plt.savefig(os.path.join(al_pm.valid_dir, "0.3_valid_E_f_H.png"))

def draw_H():
    # Residual loss & Boundary loss
    df = file_read_last_line(al_pm.H_path, type_name="float")
    fig = plt.figure()
    color_list=['#000000','#333333', '#555555', '#777777', '#999999', '#BBBBBB', '#DDDDDD', '#FFFFFF']
    j = 0
    for i in df.columns:
        if i == "type":
            continue
        plt.plot(df[i][200:], color=color_list[j],label=i)
        j = j + 1
    plt.legend()
    
    plt.ylabel("H_values")  # 设置Y轴标签
    plt.xlabel("iters")  # 设置X轴标签
    plt.savefig("RMSE")#保存图片

    # plt.show()
    plt.savefig(os.path.join(picture_save_dir, 'H.png'))

def file_read_last_line(file_path, type_name="int"):
    i = 0
    j = 0
    df = pd.DataFrame(columns=['w0', 'w1', 'w2', 'b0', 'b1', 'b2', 'type'])
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            lines = rf.readlines()  #the last line
            for line in lines:
                if j % 100 == 0:
                    line = line.replace(" ","")[1:-2].split(',')
                    line = [float(k) for k in line]
                    df.loc[i] = line
                    i = i + 1
                j = j + 1
            
    return df

def fitting_line(plt, df, columns):
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线

    sns.regplot(x=columns[0],y=columns[1], data=df, color='g', marker='.',ci=95)#, x_bins=np.arange(0,0.5,0.05), x_estimator=np.mean

    #The univariate linear regression model is established
    # from_formula = '%s~%s'%(columns[0],columns[1])
    # result = smf.ols(from_formula, data=df[columns]).fit()
    # print(result.params)
    # print(result.summary())
    # #Draw a fitting diagram.
    # y_fitted = result.fittedvalues
    # plt.plot(df[columns[0]], y_fitted, 'r-',label='OLS')
    # plt.plot(df[columns[0]], df[columns[1]], 'o', label='data')