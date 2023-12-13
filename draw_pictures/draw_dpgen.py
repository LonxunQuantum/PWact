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
from active_learning.workdir import WorkTrainDir
from active_learning.util import del_file, file_read_lines
from draw_pictures.draw_util import draw_distribution, draw_bars
import seaborn as sns
from scipy.stats import linregress
picture_save_dir = "./mlff_wu_work_dir/picture_dir"


# color_list = ["#000000", "#BDB76B", "#B8860B", "#008B8B", "#FF8C00", "#A52A2A", "#5F9EA0"]
#黑，灰色，浅棕色，浅黄色，深棕色，深黄色，深绿色
color_list = ["#000000", "#A9A9A9", "#BDB76B", "#F0E68C", "#483D8B", "#DAA520", "#008000"]
color_list = ["#008000", "#BDB76B", "#DAA520"]
# color_list = ["#BDB76B", "#008B8B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+", "o", '*']

def draw_lines(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None, \
                        xticks:list=None, xtick_loc:list=None):
    # force-kpu散点图
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 20,
        }
    plt.figure(figsize=(12,9))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    for i in range(len(y_list)):
        # if i % 2 != 0:
        #     continue
        plt.plot(x_list[i], y_list[i], \
            color=color_list[i], marker=mark_list[i], \
                label=legend_label[i])
    
    if xticks is not None:
        plt.xticks(xtick_loc, xticks, fontsize=14)
    plt.xlabel(x_label, font)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel(y_label, font)
    plt.title(title, font)
    plt.legend(fontsize=12, frameon=False, loc=location)
    plt.savefig(picture_save_path)

def draw_distribution_model_deivs():
    dir = "/share/home/wuxingxing/datas/al_dir/cu_system/dpgen_test/model_deivs"
    save_path = "/share/home/wuxingxing/datas/al_dir/cu_system/dpgen_test/model_deivs/all_model_deiv_details.png"
    details = pd.read_csv(os.path.join(dir, "all_model_deiv_details.csv"))
    draw_list = []
    for i in list(details['max_devi_f']):
        if i < 0.05:
            draw_list.append(0.25)
        elif i < 0.1:
            draw_list.append(0.1)
        elif i < 0.15:
            draw_list.append(0.15)
        elif i < 0.2:
            draw_list.append(0.2)
        elif i < 0.25:
            draw_list.append(0.25)
        elif i < 0.3:
            draw_list.append(0.3)
        else:
            draw_list.append(0.35)

    draw_distribution(draw_list, save_path, title="Force deviation of DPGEN 4 models", y_label="distribution", x_label="deviation")

def draw_bar_distribution_model_deivs():
    dir = "/share/home/wuxingxing/datas/al_dir/cu_system/dpgen_test/model_deivs"
    save_path = "/share/home/wuxingxing/datas/al_dir/cu_system/dpgen_test/model_deivs/all_model_deiv_details_bar.png"
    details = pd.read_csv(os.path.join(dir, "all_model_deiv_details.csv"))
    res = []
    x_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]#, 0.4
    x_ticks = ['< 0.05', '0.05~0.1', '0.1~0.15', '0.15~0.2', '0.2~0.25', '0.25~0.3', '>0.35']#, '> 0.3'
    for j in range(0, len(x_list)):
        draw_list = []
        for i in list(details['max_devi_f']):
            if j < 1 and i < x_list[j]:
                draw_list.append(x_list[j])
            elif j >= 1 and (i > x_list[j-1] and i < x_list[j]):
                draw_list.append(x_list[j])
        res.append(len(draw_list))

    res = [_ /sum(res) * 100 for _ in res]
    draw_bars(x_list, res, save_path=save_path, title="Force deviation of DPGEN 4 models", x_ticks = x_ticks, y_label="distribution (%)", x_label="deviation")
    #x_list, y_list, save_path, labels= None, title="bar", y_label="y1", x_label="x1"

def draw_rmse_force_in_training():
    #dpgen training loss
    dpgen_file = "/share/home/wuxingxing/datas/dpgen_al/cu_bulk_rm_05/init_train_lcurve.out"
    kpu_file = "/share/home/wuxingxing/al_dir/cu_system/iter.0029/training/model_dir/epoch_train.dat"
    #  step      rmse_trn    rmse_e_trn    rmse_f_trn         lr
    #   0      2.64e+01      1.38e+00      8.33e-01    1.0e-03
    with open(dpgen_file, 'r') as rf:
        lines = rf.readlines()
    dpgen_f_rmse = []
    for i in lines[1:]:
        step, rmse_trn, rmse_e_trn, rmse_f_trn, lr = \
                [float(_.strip()) for _ in i.split()]
        if step % 1000 == 0:
            dpgen_f_rmse.append(rmse_f_trn)
    # kpu training loss
    # epoch	 loss	 RMSE_Etot	 RMSE_Ei	 RMSE_F	 real_lr	 time
    # 1 4.815484e+03 6.938839e+01 6.425287e-01 8.572921e-01 1.000000e-03 42.88122844696045
    with open(kpu_file, 'r') as rf:
        lines = rf.readlines()
    kpu_f_rmse = []
    for idex, i in enumerate(lines[2:]):
        if idex%2 ==0:
            continue
        epoch, loss, RMSE_Etot, RMSE_Ei, RMSE_F, real_lr, time = \
                [float(_.strip()) for _ in i.split()]
        kpu_f_rmse.append(RMSE_F)
    x_list = []
    y_list = []

    x_list.append(list(range(0, len(kpu_f_rmse))))
    y_list.append(kpu_f_rmse)

    x_list.append(list(range(0, len(dpgen_f_rmse))))
    y_list.append(dpgen_f_rmse)

    save_dir = "/share/home/wuxingxing/al_dir/cu_system"
    draw_lines(x_list, y_list, ["KPU", "DPGEN"], \
    x_label = "training epochs", y_label = r"$RMSE \left(V\right)$", \
        title = "Training loss of force during the training process.", location = "best", \
            picture_save_path = os.path.join(save_dir, "training_loss_kpu_dpgen.png"))


if __name__ == "__main__":
    # draw_cu_4pahses_rmse_kpu()
    # draw_cu_4pahses_kpu_pashe()
    # draw_i0_i6_force_rmse_at_600k()
    # draw_rmse_force_in_training()
    draw_bar_distribution_model_deivs()
