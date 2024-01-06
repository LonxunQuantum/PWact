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

# color_list = ["#000000", "#BDB76B", "#B8860B", "#008B8B", "#FF8C00", "#A52A2A", "#5F9EA0"]
#黑，灰色，浅棕色，浅黄色，深棕色，深黄色，深绿色
# color_list = ["#000000", "#A9A9A9", "#BDB76B", "#F0E68C", "#483D8B", "#DAA520", "#008000"]
# color_list = ["#008000", "#BDB76B", "#DAA520"]
color_list = ["#BDB76B", "#008B8B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+", "o", '*']

'''
Description: 
draw scatter picture
Returns: 
Author: WU Xingxing
'''
def draw_scatter(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None):
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 12
        }
    font_size = 25
    plt.figure(figsize=(12,9))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid(linewidth =1.5) # 网格线
    for i in range(len(y_list)):
        # if i == 2:
        #     continue
        plt.scatter(x_list[i], y_list[i], \
            color=color_list[i], marker=mark_list[i], \
                label=legend_label[i])
    
    # plt.xticks([100,300,500,700], ["bulk", "gas", "slab", "liquid"], fontsize=25)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)   
    plt.xlabel(x_label,font, fontsize=font_size)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(y_label, font, fontsize=font_size)
    plt.title(title, font, fontsize=font_size)
    plt.legend(fontsize=font_size, frameon=False, loc=location)
    plt.savefig(picture_save_path)

def draw_lines(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None, \
                        xticks:list=None, xtick_loc:list=None):
    # force-kpu散点图
    fontsize = 25
    fontsize2 = 25
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'fontsize' : fontsize,
        }
    plt.figure(figsize=(12,10))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid(linewidth =1.5) # 网格线
    for i in range(len(y_list)):
        # if i % 2 != 0:
        #     continue
        plt.plot(x_list[i], y_list[i], \
            color=color_list[i], marker=mark_list[i], \
                label=legend_label[i], linewidth =4.0)
    
    if xticks is not None:
        plt.xticks(xtick_loc, xticks, fontsize=fontsize2)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.xlabel(x_label, font)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel(y_label, font)
    plt.title(title, font)
    plt.legend(fontsize=fontsize, frameon=False, loc=location)
    plt.savefig(picture_save_path)
    
"""
@Description :
横轴四个相名称，纵轴KPU, x_column="kpu" force KPU, "etot_kpu" energy kpu
@Returns     :
@Author       :wuxingxing
"""
def draw_cu_4pahses_kpu_pashe():
    #设置工作路径
    save_dir = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/cu_4pahses"  #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/share/home/wuxingxing/datas/system_config/cu_72104/dft_test/dft_cu_4phases/PR_bulk_gas_slab_liquid.csv" # PR_bulk_gas_slab_liquid bulk_gas_slab_liquid
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    # read data
    res = pd.read_csv(data_path, index_col=0, header=0, dtype=float)
    x_list = []
    rmse_e_list = []
    rmse_f_list = []
    kpu_f_list = []
    kpu_e_list = []
    rmse_e = np.array(res["rmse_e"])
    rmse_f = np.array(res["rmse_f"])*1000
    f_kpu = np.array(res["f_kpu"])
    etot_kpu = np.array(res["etot_kpu"])*1000
    start = 0
    for i in range(0, 4):
        x_list.append(list(range(start, start+200)))
        rmse_e_list.append(rmse_e[start:start+200])
        kpu_e_list.append(etot_kpu[start:start+200])  

        rmse_f_list.append(rmse_f[start:start+200])
        kpu_f_list.append(f_kpu[start:start+200])
        start += 200
    legend_label_list = ["bulk", "gas", "slab", "liquid"]
    # etotal-rmse-kpu散点图
    draw_scatter(kpu_e_list, rmse_e_list, legend_label_list, \
        x_label = "Energy KPU", y_label = r"$RMSE \left(meV\right)$", \
            title = "Scatter plot of \nenergy KPU versus RMSE of atom energy", location = "best", \
                picture_save_path = os.path.join(save_dir, "PRi_e_rmse_kpu_phases.png"))
    draw_scatter(kpu_f_list, rmse_f_list, legend_label_list, \
        x_label = "Force KPU", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
            title = "Scatter plot of \nforce KPU versus RMSE of atom force", location = "best", \
                picture_save_path = os.path.join(save_dir, "PRi_force_rmse_kpu_phases.png"))
    print()

def draw_i0_i6_force_rmse_at_600k():
    iters = ["iter.0000", "iter.0001", "iter.0002", "iter.0003", "iter.0004", "iter.0005", "iter.0006", "iter.0007"]
    dir = "/share/home/wuxingxing/al_dir/cu_system/{}/training/model_dir/pbe_dft_600k_valid"

    save_dir = "/share/home/wuxingxing/al_dir/cu_system/i0_i6_dft_kpu_test_final"
    # force_rmse = []
    # xrange = []
    # for i, iter in enumerate(iters):
    #     path = os.path.join(dir.format(iter), "prediction.csv")
    #     detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
    #     detail.sort_values(by="img_idx", inplace=True, ascending=True)
    #     x_indexs = list(range(0, detail.shape[0], 10))
    #     detail = detail.loc[detail['img_idx'].isin(x_indexs)]
    #     force_rmse.append(detail['force_rmse']*1000)
    #     xrange.append(detail['img_idx'])
    # draw_lines(xrange, force_rmse, iters, \
    # x_label = "AIMD trajectory of Copper Bulk system at 600K", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
    #     title = "Force RMSE of model which training data from KPU method at valid set", location = "best", \
    #         picture_save_path = os.path.join(save_dir, "force_rmse_i0_i6_600k_kpu_data.png"))

    rmse_force_avg = []
    rmse_etot_avg = []
    for i, iter in enumerate(iters):
        path = os.path.join(dir.format(iter), "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        rmse_force_avg.append(detail['force_rmse'].mean()*1000)
        rmse_etot_avg.append(detail['etot_atom_rmse'].mean()*1000)
    xrange = list(range(0, len(rmse_force_avg), 1))
    legend_label = [""]
    xtick = ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7"]
    draw_lines([xrange], [rmse_force_avg], legend_label, \
    x_label = "Iters of active learning", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "Average force RMSE of model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_avg_force_rmse_i0_i6_600k_kpu_data.png"),
            xticks=xtick, xtick_loc=xrange)
    
    draw_lines([xrange], [rmse_etot_avg], legend_label,\
    x_label = "Iters of active learning", y_label = r"$RMSE \left(meV\right)$", \
        title = "Average atom energy RMSE of model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_avg_energy_rmse_i0_i6_600k_kpu_data.png"),
            xticks=xtick, xtick_loc=xrange)
    
def draw_i0_i6_force_rmse_kpu_rand_at_600k():
    # 不加入预训练数据KPU结果要好于random，加入之后结果能量上差不多，力场KPU稍微好些
    # 后缀为dpall_retrain的表示使用全部数据重新训练了模型，结果要好于后缀dpall 即iter.07开始时候的模型
    types = ["kpu"]#, "random"
    # data without pre-training
    # dirs = ["/share/home/wuxingxing/datas/al_dir/cu_system/i0_i6_dft_kpu_test_final/final_test/rm5/pbe_dft_600k_kpu_data_valid",
    #         "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i6/rand_data/pbe_dft_600k_kpu_data_valid"]

    # data with pre-training  
    # dirs = ["/share/home/wuxingxing/datas/al_dir/cu_system/iter.0007/training/model_dir/pbe_dft_600k_valid",
    #         "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i6/rand_data/pbe_dft_600k_kpu_data_valid"]

    # data with pre-training
    dirs = ["/share/home/wuxingxing/datas/al_dir/cu_system/kpu_test_i6/pbe_valid_600k_all"]
    #"/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i6/rand_data/pbe_dft_600k_kpu_data_valid"
    save_dir = "/share/home/wuxingxing/al_dir/cu_system/i0_i6_dft_kpu_test_final"

    force_rmse = []
    etot_rmse = []
    xrange = []
    for i, dir in enumerate(dirs):
        path = os.path.join(dir, "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        x_indexs = list(range(0, detail.shape[0], 10))
        detail = detail.loc[detail['img_idx'].isin(x_indexs)]
        force_rmse.append(detail['force_rmse']*1000)
        etot_rmse.append(detail["etot_atom_rmse"]*1000)
        xrange.append(detail['img_idx'])

    # dpgen = "/share/home/wuxingxing/datas/al_dir/cu_system/dpgen_test/detail_i0_i6_600k_rmse_mcloud.csv"
    # dpgen = "/share/home/wuxingxing/datas/dpgen_al/cu_bulk_rm_05/i6_dpgen_all_data_details.csv"
    dpgen = "/share/home/wuxingxing/datas/al_dir/cu_system/i0_i6_dft_kpu_test/pbe_600k_details.csv"
    detail = pd.read_csv(dpgen, index_col=0, header=0, dtype=float)
    detail = detail.loc[detail.index.isin(x_indexs)]
    force_rmse.append(detail['rmse_f']*1000)
    etot_rmse.append(detail['rmse_e']*1000)
    xrange.append(xrange[0])
    types.append("dpgen")
    draw_lines(xrange, force_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 600K (fs)", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "Force RMSE of \nKPU-model and DPGEN-model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_force_rmse_kpu_dpgen_i0_i6_600k_kpu_data_dpall_retrain.png"))

    draw_lines(xrange, etot_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 600K (fs)", y_label = r"$RMSE \left(meV\right)$", \
        title = "Per atom energe RMSE \nof KPU-model \nand DPGEN-model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_atom_e_rmse_kpu_dpgen_i0_i6_600k_kpu_data_dpall_retrain.png"))

def draw_iters_force_rmse_at_1400k():
    iters = ["iter.0019","iter.0020","iter.0021","iter.0022","iter.0023","iter.0024","iter.0026","iter.0027","iter.0028"]
    dir = "/share/home/wuxingxing/al_dir/cu_system/{}/training/model_dir/pbe_dft_1400k_kpu_data_valid_1400k_all"

    save_dir = "/share/home/wuxingxing/al_dir/cu_system/kpu_test_i28"

    rmse_force_avg = []
    rmse_etot_avg = []
    for i, iter in enumerate(iters):
        path = os.path.join(dir.format(iter), "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        rmse_force_avg.append(detail['force_rmse'].mean()*1000)
        rmse_etot_avg.append(detail['etot_atom_rmse'].mean()*1000)
    xrange = list(range(0, len(rmse_force_avg), 1))
    legend_label = [""]
    xtick = ["i19","i20","i21","i22","i23","i24","i25","i26","i27"]
    draw_lines([xrange], [rmse_force_avg], legend_label, \
    x_label = "Iters of active learning", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "Average force RMSE of model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_avg_force_rmse_iter_1400k_kpu_data.png"),
            xticks=xtick, xtick_loc=xrange)
    
    draw_lines([xrange], [rmse_etot_avg], legend_label,\
    x_label = "Iters of active learning", y_label = r"$RMSE \left(meV\right)$", \
        title = "Average atom energy RMSE of model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_avg_energy_rmse_iter_1400k_kpu_data.png"),
            xticks=xtick, xtick_loc=xrange)
    

def draw_force_rmse_kpu_rand_at_1400k():
    types = ["kpu"]#, "random"

    #/share/home/wuxingxing/al_dir/cu_system/iter.0029/training/model_dir
    #/share/home/wuxingxing/al_dir/cu_system/kpu_test_i28
    dirs = ["/share/home/wuxingxing/al_dir/cu_system/iter.0028/training/model_dir/pbe_dft_1400k_kpu_data_valid_1400k_all",
            ] # kpu_data_only1400k
    #"/share/home/wuxingxing/al_dir/cu_system/rand_test_i28/rand_data/pbe_dft_1400k_rand_data_valid"


    save_dir = "/share/home/wuxingxing/al_dir/cu_system/kpu_test_i28/kpu_test_i28_final"

    force_rmse = []
    etot_rmse = []
    xrange = []
    for i, dir in enumerate(dirs):
        path = os.path.join(dir, "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        x_indexs = list(range(0, detail.shape[0], 10))
        detail = detail.loc[detail['img_idx'].isin(x_indexs)]
        force_rmse.append(detail['force_rmse']*1000)
        etot_rmse.append(detail["etot_atom_rmse"]*1000)
        xrange.append(detail['img_idx'])

    dpgen = "/share/home/wuxingxing/datas/al_dir/cu_system/kpu_test_i28/pbe_detail_1400k_0s.csv"
    detail = pd.read_csv(dpgen, index_col=0, header=0, dtype=float)
    detail = detail.loc[detail.index.isin(x_indexs)]
    force_rmse.append(detail['rmse_f']*1000)
    etot_rmse.append(detail['rmse_e']*1000)
    xrange.append(xrange[0])
    types.append("dpgen")
    draw_lines(xrange, force_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 1400K (fs)", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "Force RMSE of \nKPU-model and DPGEN-model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_force_rmse_kpu_dpgen_1400k_all_kpu_data.png"))

    draw_lines(xrange, etot_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 1400K (fs)", y_label = r"$RMSE \left(meV\right)$", \
        title = "Per atom energe RMSE of \nKPU-model and DPGEN-model at valid set", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_atom_e_rmse_kpu_dpgen_1400k_all_kpu_data.png"))

def draw_time():
    dpgen_time = "/share/home/wuxingxing/al_dir/cu_system/dpgen_test/all_data/lcurve.out"
    kpu_time  = "/share/home/wuxingxing/al_dir/cu_system/dpgen_test/all_data/kpu_time.txt"

    kpu = np.loadtxt(kpu_time)
    mse_e = kpu[:,0]*1000
    mse_f = kpu[:,1]*1000
    time = list(np.array(kpu[:,2])/60)
    real_time = [time[0]]
    for i in time[1:]:
        real_time.append(real_time[-1]+i)

    with open(dpgen_time, "r") as rf:
            lines = rf.readlines()
    # "step      rmse_trn    rmse_e_trn    rmse_f_trn         lr"  
    rmse_e_trn = []
    rmse_f_trn = []
    time_dpgen = []
    base_time = 15497/400000
    for i in lines[1:]:
        i = i.strip()
        step, rmse_trn,  rmse_e_, rmse_f_, lr = \
            [float(_.strip()) for _ in i.split()]
        if step % 8000 == 0 :
            rmse_f_trn.append(rmse_f_)
            rmse_e_trn.append(rmse_e_)
            time_dpgen.append(base_time*step/60)


    xrange = []
    xrange.append(time_dpgen)
    xrange.append(real_time)
    etot_rmse=[]
    etot_rmse.append(np.array(rmse_e_trn)*1000)
    etot_rmse.append(np.array(mse_e)/108)
    force_rmse=[]
    force_rmse.append(np.array(rmse_f_trn)*1000)
    force_rmse.append(mse_f)

    types = ["dpgen", "kpu"]
    save_dir = "/share/home/wuxingxing/al_dir/cu_system/dpgen_test/all_data/"
    draw_lines(xrange, etot_rmse, types, \
    x_label = "train time (min)", y_label = r"$RMSE \left(meV\right)$", \
        title = "The trend of the training energy loss\n with training time increasing", location = "best", \
            picture_save_path = os.path.join(save_dir, "train_time_compare_etot.png"))

    draw_lines(xrange, force_rmse, types, \
    x_label = "train time (min)", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "The trend of the training force loss\n with training time increasing", location = "best", \
            picture_save_path = os.path.join(save_dir, "train_time_compare_force.png"))
       
    
def draw_force_rmse_kpu():
    types = ["1400K"]#, "600K",  "/share/home/wuxingxing/datas/al_dir/cu_system/kpu_test_i6/pbe_valid_600k_all",
    dirs = [
        "/share/home/wuxingxing/al_dir/cu_system/iter.0028/training/model_dir/pbe_dft_1400k_kpu_data_valid_1400k_all"
        ]
    #"/share/home/wuxingxing/al_dir/cu_system/iter.0028/training/model_dir/pbe_dft_1400k_kpu_data_valid_1400k_all"
    save_dir = "/share/home/wuxingxing/al_dir/cu_system/kpu_test_i28/kpu_test_i28_final"
    force_rmse = []
    etot_rmse = []
    xrange = []
    for i, dir in enumerate(dirs):
        path = os.path.join(dir, "prediction.csv")
        detail = pd.read_csv(path, index_col=0, header=0, dtype=float)
        detail.sort_values(by="img_idx", inplace=True, ascending=True)
        x_indexs = list(range(0, detail.shape[0], 10))
        detail = detail.loc[detail['img_idx'].isin(x_indexs)]
        force_rmse.append(detail['force_rmse']*1000)
        etot_rmse.append(detail["etot_atom_rmse"]*1000)
        xrange.append(detail['img_idx'])

    draw_lines(xrange, force_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 1400K (fs)", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
        title = "Force RMSE of KPU-model \nat valid set from 1400K AIMD", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_force_rmse_kpu_all_kpu_data_1400k.png"))

    draw_lines(xrange, etot_rmse, types, \
    x_label = "AIMD trajectory of Copper Bulk system at 1400K (fs)", y_label = r"$RMSE \left(meV\right)$", \
        title = "Per atom energe RMSE of KPU-model \nat valid set from 1400K AIMD", location = "best", \
            picture_save_path = os.path.join(save_dir, "pbe_atom_e_rmse_kpu_all_kpu_data_1400k.png")) 

    # draw_lines(xrange, force_rmse, types, \
    # x_label = "AIMD trajectory of Copper Bulk system at 600K (fs)", y_label = r"$RMSE \left(meV/\overset{o}{A}\right)$", \
    #     title = "Force RMSE of KPU-model \nat valid set from 600K AIMD", location = "best", \
    #         picture_save_path = os.path.join(save_dir, "pbe_force_rmse_kpu_all_kpu_data_600k.png"))

    # draw_lines(xrange, etot_rmse, types, \
    # x_label = "AIMD trajectory of Copper Bulk system at 600K (fs)", y_label = r"$RMSE \left(meV\right)$", \
    #     title = "Per atom energe RMSE of KPU-model \nat valid set from 600K AIMD", location = "best", \
    #         picture_save_path = os.path.join(save_dir, "pbe_atom_e_rmse_kpu_all_kpu_data_600k.png")) 
       
            
if __name__ == "__main__":
    # draw_cu_4pahses_rmse_kpu()
    draw_cu_4pahses_kpu_pashe()
    # draw_i0_i6_force_rmse_at_600k()
    # draw_i0_i6_force_rmse_kpu_rand_at_600k()
    # draw_iters_force_rmse_at_1400k()
    # draw_force_rmse_kpu_rand_at_1400k()
    # draw_time()
    # draw_force_rmse_kpu()