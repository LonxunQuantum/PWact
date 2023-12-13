'''
Author: starsparkling stars_sparkling@163.com
Date: 2022-10-12 16:19:00
LastEditors: Please set LastEditors
LastEditTime: 2023-04-22 22:04:52
FilePath: /MLFF_wu_dev/draw_pictures/draw_util.py
Description: @_@
symbol__custom_string_obkoro1: 
symbol__custom_string_obkoro1_copyright: Copyright (c) ${now_year} by ${git_name_email}, All Rights Reserved. 
'''
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

color_list = ["#000000", "#008B8B", "#BDB76B", "#B8860B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+"]

def draw_distribution(dist, save_path, title="distribution", y_label="times", x_label="kpu range"):
    plt.figure(dpi=120)
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})#修改背景色
    g=sns.distplot(dist,
                # hist=True,#默认绘制直方图，详细参考plt.hist norm_hist=True
                norm_hist=True,
                kde=False,
                color="#098154")#修改柱子颜色
    plt.ylabel(y_label)  # 设置Y轴标签
    plt.xlabel(x_label)  # 设置X轴标签
    plt.title(title)
    plt.savefig(save_path)

'''
description: draw a picture of lines sucha as y1 = αx, y2 = βx
param {*} x_list
param {*} y_list
param {*} save_path
param {*} labels
param {*} title
param {*} y_label
param {*} x_label
return {*}
'''
def draw_lines(x_list, y_list, save_path, labels= None, title="lines", y_label="y1", x_label="x1"):
    plt.figure(dpi=120)
    for i in range(len(y_list)):
        _label = "" if labels is None else labels[i]
        plt.plot(x_list[i], y_list[i], 'g-', label=_label)
    plt.grid()
    plt.ylabel(y_label)  # 设置Y轴标签
    plt.xlabel(x_label)  # 设置X轴标签
    plt.title(title)
    plt.savefig(save_path)

def draw_bars(x_list, y_list, save_path, labels= None, title="bar", x_ticks = [], y_label="y1", x_label="x1"):
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 12,
        'fontsize' : 25
        }
    # 中文显示问题
    font_size = 25
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 25
    # 设置图大小
    plt.figure(figsize=(20,8))

    x = [_*2 for _ in list(range(0, len(x_list)))]
    # x = list(x_list) # 获取x轴数据(字典的键)
    y = list(y_list) # 获取y轴数据(字典的值)

    plt.bar(x,y,width=0.75,bottom=0,align='edge',color='#D3D3D3',edgecolor ='black',linewidth=2)
    
    x = [_ + 0.25 for _ in x]
    plt.xticks(x, x_ticks, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(linewidth =1.5)
    # 绘制标题
    plt.title(title, fontsize=25)

    # 设置轴标签
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    plt.savefig(save_path)
# if __name__ == "__main__":
