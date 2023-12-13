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
import seaborn as sns
from scipy.stats import linregress
picture_save_dir = "./mlff_wu_work_dir/picture_dir"

# color_list = ["#000000", "#BDB76B", "#B8860B", "#008B8B", "#FF8C00", "#A52A2A", "#5F9EA0"]
#黑，灰色，浅棕色，浅黄色，深棕色，深黄色，深绿色
# color_list = ["#000000", "#A9A9A9", "#BDB76B", "#F0E68C", "#483D8B", "#DAA520", "#008000"]
# color_list = ["#008000", "#BDB76B", "#DAA520"]
color_list = ["#008000", "#800000", "#FF8C00", "#A52A2A"]
mark_list = ["s", "^", "v", "^", "+", "", '*']

def draw_lines(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None, \
                        xticks:list=None, xtick_loc:list=None, withmark=True, withxlim=True):
    # force-kpu散点图
    fontsize = 35
    fontsize2 = 35
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'fontsize' : fontsize,
        }
    plt.figure(figsize=(12,10))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    for i in range(len(y_list)):
        if withmark:
            plt.plot(x_list[i], y_list[i], \
                color=color_list[i], marker=mark_list[i], markersize=8, \
                    label=legend_label[i], linewidth =3.0)
        else:
            plt.plot(x_list[i], y_list[i], \
                color=color_list[i], \
                    label=legend_label[i], linewidth =3.0)
                   
    if xticks is not None:
        plt.xticks(xtick_loc, xticks, fontsize=fontsize2)
    if withxlim is True:
        plt.xlim(left=0, right=max(x_list[0]))
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.xlabel(x_label, font)
    plt.yscale('log')
    plt.grid(linewidth =1.5) # 网格线
    # plt.xscale('log')
    plt.ylabel(y_label, font)
    plt.title(title, font)
    plt.legend(fontsize=fontsize, frameon=False, loc=location)
    plt.tight_layout()
    plt.savefig(picture_save_path)
    
    
def draw_rmse_kpu_at_800k():
    types = ["Si"]#, "random"
    dirs = ["/data/home/wuxingxing/al_dir/si_2/iter.0010/training/model_dir/800k_valid"]
    save_dir = "/data/home/wuxingxing/al_dir/si_2/valid"

    force_rmse = []
    etot_rmse = []
    xrange = []
    for i, dir in enumerate(dirs):
        path = os.path.join(dir, "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        x_indexs = list(range(0, detail.shape[0], 10))
        detail = detail.loc[detail['img_idx'].isin(x_indexs)]
        force_rmse.append(detail['force_rmse'])
        etot_rmse.append(detail["etot_atom_rmse"])
        xrange.append(detail['img_idx'])

    xrange.append(xrange[0])

    xtick_loc = list(range(0, 1001, 200))
    xticks = xtick_loc
    draw_lines(xrange, force_rmse, types, \
    x_label = "Index of AIMD trajectory", y_label = r"RMSE $\mathrm{(eV/\overset{o}{A})}$", \
        title = "Force RMSE", location = "upper right", \
            xtick_loc=xtick_loc, xticks=xticks,\
            picture_save_path = os.path.join(save_dir, "si_800k_valid_force_rmse.png"), withmark=False, withxlim=False)

    draw_lines(xrange, etot_rmse, types, \
    x_label = "Index of AIMD trajectory", y_label = r"RMSE $\left(\mathrm{eV}\right)$", \
        title = "Energy RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_800k_valid_energy_rmse.png"), withmark=False, withxlim=False)


def draw_rmse_kpu_at_1100k():
    types = ["Si"]#, "random"
    dirs = ["/data/home/wuxingxing/al_dir/si_2/iter.0010/training/model_dir/1100k_valid"]
    save_dir = "/data/home/wuxingxing/al_dir/si_2/valid"

    force_rmse = []
    etot_rmse = []
    xrange = []
    for i, dir in enumerate(dirs):
        path = os.path.join(dir, "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        x_indexs = list(range(0, detail.shape[0], 10))
        detail = detail.loc[detail['img_idx'].isin(x_indexs)]
        force_rmse.append(detail['force_rmse'])
        etot_rmse.append(detail["etot_atom_rmse"])
        xrange.append(detail['img_idx'])

    xrange.append(xrange[0])

    xtick_loc = list(range(0, 1001, 200))
    xticks = xtick_loc
    draw_lines(xrange, force_rmse, types, \
    x_label = "Index of AIMD trajectory", y_label = r"RMSE $\mathrm{(eV/\overset{o}{A})}$", \
        title = "Force RMSE", location = "upper right", \
            xtick_loc=xtick_loc, xticks=xticks,\
            picture_save_path = os.path.join(save_dir, "si_1100k_valid_force_rmse.png"), withmark=False, withxlim=False)

    draw_lines(xrange, etot_rmse, types, \
    x_label = "Index of AIMD trajectory", y_label = r"RMSE $\left(\mathrm{eV}\right)$", \
        title = "Energy RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_1100k_valid_energy_rmse.png"), withmark=False, withxlim=False)

def draw_iters_force_rmse_at_600k_kpu():
    iters = ["iter.0000", "iter.0001", "iter.0005", "iter.0006",\
              "iter.0007", "iter.0008", "iter.0009"]
    types = "800k_valid" #"1400k_valid"
    dir = "/data/home/wuxingxing/al_dir/si_2/{}/training/model_dir/{}"

    save_dir = "/data/home/wuxingxing/al_dir/si_2/valid"

    rmse_force_avg = []
    rmse_etot_avg = []
    for i, iter in enumerate(iters):
        path = os.path.join(dir.format(iter, types), "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        rmse_force_avg.append(detail['force_rmse'].mean())
        rmse_etot_avg.append(detail['etot_atom_rmse'].mean())

    xrange = list(range(0, len(rmse_force_avg), 1))
    
    legend_label = ["Si"]

    xtick = ["0", "1", "5", "6", "7", "8", "9"]
    
    draw_lines([xrange], [rmse_force_avg], legend_label, \
    x_label = "Iteration of active learning", y_label = r"$RMSE \left(eV/\overset{o}{A}\right)$", \
        title = "Force RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_800k_iters_kpu_force_rmse.png"),
            xticks=xtick, xtick_loc=xrange)
    
    draw_lines([xrange,xrange], [rmse_etot_avg], legend_label, \
    x_label = "Iteration of active learning", y_label = r"$RMSE \left(eV\right)$", \
        title = "Energy RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_800k_iters_kpu_energy_rmse.png"),
            xticks=xtick, xtick_loc=xrange)

def draw_iters_force_rmse_at_800k_kpu():
    iters = ["iter.0000", "iter.0001", "iter.0005", "iter.0006",\
              "iter.0007", "iter.0008", "iter.0009"]
    types = "1100k_valid" #"1400k_valid"
    dir = "/data/home/wuxingxing/al_dir/si_2/{}/training/model_dir/{}"

    save_dir = "/data/home/wuxingxing/al_dir/si_2/valid"

    rmse_force_avg = []
    rmse_etot_avg = []
    for i, iter in enumerate(iters):
        path = os.path.join(dir.format(iter, types), "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        rmse_force_avg.append(detail['force_rmse'].mean())
        rmse_etot_avg.append(detail['etot_atom_rmse'].mean())

    xrange = list(range(0, len(rmse_force_avg), 1))
    
    legend_label = ["Si"]

    xtick = ["0", "1", "5", "6", "7", "8", "9"]
    
    draw_lines([xrange], [rmse_force_avg], legend_label, \
    x_label = "Iteration of active learning", y_label = r"$RMSE \left(eV/\overset{o}{A}\right)$", \
        title = "Force RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_1100k_iters_kpu_force_rmse.png"),
            xticks=xtick, xtick_loc=xrange)
    
    draw_lines([xrange,xrange], [rmse_etot_avg], legend_label, \
    x_label = "Iteration of active learning", y_label = r"$RMSE \left(eV\right)$", \
        title = "Energy RMSE", location = "upper right", \
            picture_save_path = os.path.join(save_dir, "si_1100k_iters_kpu_energy_rmse.png"),
            xticks=xtick, xtick_loc=xrange)
    
if __name__ == "__main__":
    draw_rmse_kpu_at_800k()
    draw_rmse_kpu_at_1100k()

    draw_iters_force_rmse_at_600k_kpu()
    draw_iters_force_rmse_at_800k_kpu()
    