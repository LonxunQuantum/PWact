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

color_list = ["#000000", "#008B8B", "#BDB76B", "#B8860B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+"]
split_list = ["0.05", "0.1", "0.15", "0.2", "uper"]

def get_image_index(dir_path):
    file_list = os.listdir(dir_path)
    image_index = []
    for f in file_list:
        image_index.append(int(f[6:-4]))
    return image_index

def draw_force_kpu_image():
    #read batch index and etot kpu and etot_loss
    batch_index = file_read_lines(os.path.join(valid_dir, "kpu_valid_batch.dat"), "int")
    #columns=["batch", "etot_lab", "etot_pre", "etot_loss", "kpu_etot", "f_lab", "f_pre", "f_loss", "f_kpu"]
    valid_loss = pd.read_csv(os.path.join(valid_dir, "valid_kpu.csv"), index_col=0, header=0, dtype=float)
    # etot_data = file_read_lines(valid_kpu_etot_path, "float")#渣的数据结构，操作不方便
    image_index = get_image_index(os.path.join(valid_dir, "kpu_force"))
    # batch_index = batch_index[(len(batch_index)-image_index.__len__()):]
    valid_loss.sort_values(by="f_loss", inplace=True, ascending=True)
    lay_loss = {0:[],1:[],2:[],3:[],4:[]}
    for i in range(valid_loss.shape[0]):
        if valid_loss['f_loss'][i] <= 0.05:
            lay_loss[0].append(i)
        elif valid_loss['f_loss'][i] <= 0.1:
            lay_loss[1].append(i)
        elif valid_loss['f_loss'][i] <= 0.15:
            lay_loss[2].append(i)
        elif valid_loss['f_loss'][i] <= 0.2:
            lay_loss[3].append(i)
        else:
            lay_loss[4].append(i)
    #散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    for i in range(5):
        plt.scatter(valid_loss.loc[lay_loss[i]]["f_kpu"],valid_loss.loc[lay_loss[i]]["f_loss"],color=color_list[i],marker=mark_list[i], label="f_loss < {} with {} images".format(split_list[i], len(lay_loss[i])))
    #线性回归
    slope, intercept, r, p_value, std_err = linregress(valid_loss["f_kpu"], valid_loss["f_loss"])
    ax_b = [slope, intercept]
    slope = str(np.round(ax_b[0], 8))
    intercept = str(np.round(ax_b[1], 8))
    eqn = 'F_kpu LstSQ: y = {}x + {}'.format(slope, intercept)
    plt.plot(valid_loss["f_kpu"], ax_b[0] * valid_loss["f_kpu"] + ax_b[1], 'r-', label=eqn)
    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    plt.xlabel("forece avg-kpu of images")
    plt.ylabel("Force_MAE")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(valid_dir, "image_force_kpu.png"))

    #能量散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    for i in range(5):
        plt.scatter(valid_loss.loc[lay_loss[i]]["kpu_etot"],valid_loss.loc[lay_loss[i]]["etot_loss"],color=color_list[i],marker=mark_list[i], label="f_loss < {} with {} images".format(split_list[i], len(lay_loss[i])))
    
    #能量线性回归
    slope, intercept, r, p_value, std_err = linregress(valid_loss["kpu_etot"], valid_loss["etot_loss"])
    ax_b = [slope, intercept]
    slope = str(np.round(ax_b[0], 8))
    intercept = str(np.round(ax_b[1], 8))
    eqn = 'Etotal_kpu LstSQ: y = {}x + {}'.format(slope, intercept)
    plt.plot(valid_loss["kpu_etot"], ax_b[0] * valid_loss["kpu_etot"] + ax_b[1], 'r-', label=eqn)

    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    plt.xlabel("kpu")
    plt.ylabel("Etotal_MAE avg-kpu of image")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(valid_dir, "image_etot_kpu.png"))

def draw_force_kpu_atoms():
    valid_loss = pd.read_csv(os.path.join(valid_dir, "valid_kpu.csv"),  index_col=0, header=0, dtype=float)
    # etot_data = file_read_lines(valid_kpu_etot_path, "float")#渣的数据结构，操作不方便
    image_index = get_image_index(os.path.join(valid_dir, "kpu_force"))
    # batch_index = batch_index[(len(batch_index)-image_index.__len__()):]
    valid_loss.sort_values(by="f_loss", inplace=True, ascending=True)
    lay_loss = {0:[],1:[],2:[],3:[],4:[]}
    for i in range(valid_loss.shape[0]):
        if valid_loss['f_loss'][i] <= 0.05:
            lay_loss[0].append(i)
        elif valid_loss['f_loss'][i] <= 0.1:
            lay_loss[1].append(i)
        elif valid_loss['f_loss'][i] <= 0.15:
            lay_loss[2].append(i)
        elif valid_loss['f_loss'][i] <= 0.2:
            lay_loss[3].append(i)
        else:
            lay_loss[4].append(i)
    
    column_name=["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    lay_atom_loss_df = {0:pd.DataFrame(columns=column_name),1:pd.DataFrame(columns=column_name),\
                        2:pd.DataFrame(columns=column_name),3:pd.DataFrame(columns=column_name),\
                            4:pd.DataFrame(columns=column_name)}

    for i in range(valid_loss.shape[0]):
        force_kpu = pd.read_csv(os.path.join(valid_dir, "kpu_force/image_%d.csv"%i), index_col=0, header=0, dtype=float)
        force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
        force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
        if i in lay_loss[0]:
            lay_atom_loss_df[0] = pd.concat([lay_atom_loss_df[0],force_kpu])
        elif i in lay_loss[1]:
            lay_atom_loss_df[1] = pd.concat([lay_atom_loss_df[1],force_kpu])
        elif i in lay_loss[2]:
            lay_atom_loss_df[2] = pd.concat([lay_atom_loss_df[2],force_kpu])
        elif i in lay_loss[3]:
            lay_atom_loss_df[3] = pd.concat([lay_atom_loss_df[3],force_kpu])
        else:
            lay_atom_loss_df[4] = pd.concat([lay_atom_loss_df[4],force_kpu])
    
    # 力散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    for i in range(5):
        plt.scatter(lay_atom_loss_df[i]["kpu"], lay_atom_loss_df[i]["loss"], color=color_list[i], marker=mark_list[i], label="f_loss < {} with {} images".format(split_list[i], len(lay_loss[i])))
    #线性回归
    # df = df.reset_index()
    df = pd.DataFrame(columns=lay_atom_loss_df[0].columns.values)
    df = pd.concat([df, lay_atom_loss_df[0], lay_atom_loss_df[1], lay_atom_loss_df[2], lay_atom_loss_df[3], lay_atom_loss_df[4]])
    slope, intercept, r, p_value, std_err = linregress(df["kpu"].astype(float), df["loss"].astype(float))
    ax_b = [slope, intercept]
    slope = str(np.round(ax_b[0], 8))
    intercept = str(np.round(ax_b[1], 8))
    eqn = 'atom forece kpu LstSQ: y=' + slope + 'x' + intercept
    plt.plot(df["kpu"], ax_b[0] * df["kpu"] + ax_b[1], 'r-', label=eqn)

    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    plt.xlabel("forece kpu of atoms")
    plt.ylabel("Force_MAE")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(valid_dir, "atom_force_kpu.png"))

def draw_kpu():
    # draw_force_kpu_atoms()
    column_name=["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    avg_column = ["loss", "kpu"]

    system_dict = {"cuo":[29, 8], "cuc":[6, 29], "li":[3]} # cu_4phase #cu_bulk #cuo
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system".format(i) for i in list(system_dict.keys())]   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    train_type = ["init_drop_10pct_dpnn", "init_drop_20pct_dpnn", "init_drop_100pct_dpnn"]#
    labels = ["10%_dpkf", "20%_dpkf", "100%_dpkf"]
    sys_index = 0
    for root_dir in work_root_dir:
        kpu_dict = {}
        kpu_avg_dict = {}
        train_type_index = 0
        for t_type in train_type:
            work_dir = WorkTrainDir(root_dir, t_type)
            # image_index = get_image_index(os.path.join(work_dir.log_dir, "kpu_dir"))
            file_list = os.listdir(os.path.join(work_dir.log_dir, "kpu_dir"))
            all_images_kpu = pd.DataFrame(columns=column_name)
            avg_images_kpu = pd.DataFrame(columns=avg_column)
            for file in file_list:
                if "image" not in file:
                    continue
                force_kpu = pd.read_csv(os.path.join(work_dir.log_dir, "kpu_dir/{}".format(file)), index_col=0, header=0, dtype=float)
                force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                all_images_kpu = pd.concat([all_images_kpu,force_kpu])
                avg_images_kpu.loc[int(file[6:-4])] = [force_kpu["loss"].mean(), force_kpu["kpu"].mean()]

            kpu_avg_dict[labels[train_type_index]] = avg_images_kpu
            kpu_dict[labels[train_type_index]] = all_images_kpu
            train_type_index = train_type_index + 1
            
        # 力散点图
        plt.figure(figsize=(18,12))
        plt.style.use('classic') # 画板主题风格
        plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
        plt.grid() # 网格线
        for i in labels:
            plt.scatter(kpu_dict[i]["kpu"], kpu_dict[i]["loss"], color=color_list[labels.index(i)], marker=mark_list[labels.index(i)], label=i)
        #线性回归
        # df = df.reset_index()
        # df = pd.DataFrame(columns=kpu_dict[0].columns.values)
        # df = pd.concat([df, kpu_dict[0], kpu_dict[1], kpu_dict[2], lay_atom_loss_df[3], lay_atom_loss_df[4]])
        # slope, intercept, r, p_value, std_err = linregress(df["kpu"].astype(float), df["loss"].astype(float))
        # ax_b = [slope, intercept]
        # slope = str(np.round(ax_b[0], 8))
        # intercept = str(np.round(ax_b[1], 8))
        # eqn = 'atom forece kpu LstSQ: y=' + slope + 'x' + intercept
        # plt.plot(df["kpu"], ax_b[0] * df["kpu"] + ax_b[1], 'r-', label=eqn)

        # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
        plt.xlabel("forece kpu of atoms")
        plt.ylabel("Force_MAE")
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(work_dir.work_dir, "{}_force_kpu.png".format(list(system_dict.keys())[sys_index])))
        sys_index = sys_index + 1 

        # avg 力散点图
        plt.figure(figsize=(18,12))
        plt.style.use('classic') # 画板主题风格
        plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
        plt.grid() # 网格线
        for i in labels:
            plt.scatter(kpu_avg_dict[i]["kpu"], kpu_avg_dict[i]["loss"], color=color_list[labels.index(i)], marker=mark_list[labels.index(i)], label=i)
        #线性回归
        plt.xlabel("forece kpu of atoms")
        plt.ylabel("Force_MAE")
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(work_dir.work_dir, "{}_force_avg_kpu.png".format(list(system_dict.keys())[sys_index])))
        sys_index = sys_index + 1 

"""
@Description :
横轴KPU, x_column="kpu" force KPU, "etot_kpu" energy kpu 纵轴 rmse
@Returns     :
@Author       :wuxingxing
"""


def draw_cu_4pahses_rmse_kpu():
    def draw_scatter(train_type, kpu_dict, x_column, y_column, x_label, y_label, title, location, picture_save_path):
            # force-kpu散点图
            font =  {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 20,
            }
            plt.figure(figsize=(12,9))
            plt.style.use('classic') # 画板主题风格
            plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
            plt.grid() # 网格线
            for i in range(len(train_type)):
                kpu_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][x_column].min(),4),round(kpu_dict[train_type[i]][x_column].max(),4))
                y_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][y_column].min(),4),round(kpu_dict[train_type[i]][y_column].max(),4))
                legend_label = "{} with {} images, KPU values {}, {} values {}".format(train_type[i], kpu_dict[train_type[i]].shape[0], kpu_range, y_label, y_range)
                # y_kpu = [math.log10(k) for k in (list(kpu_dict[train_type[i]][x_column]))]
                print(title, ", ", legend_label)
                plt.scatter(kpu_dict[train_type[i]][x_column], kpu_dict[train_type[i]][y_column], \
                    color=color_list[i], marker=mark_list[i], \
                        label=legend_label)
            plt.xscale('log')
            plt.xlabel(x_label, font)
            plt.ylabel(y_label, font)
            plt.title(title, font)
            plt.legend(fontsize=12, frameon=False, loc=location)
            # plt.show()
            plt.savefig(picture_save_path)

    #设置工作路径
    system = "cu_4phases" # cu_4phase #cu_bulk #cuo
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system".format(system)]   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/data/data/wuxingxing/datas/{}/train_data".format(system)
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    train_type = ["bulk", "slab", "liquid", "gas"]# 
    config_dict = {"bulk":['MOVEMENT1000K-b','MOVEMENT1000-300K-b'], "slab":['MOVEMENT1000K-slab', 'MOVEMENT1300K-slab', 'MOVEMENT1500K-slab'], "liquid":['MOVEMENT2000K-l'], "gas":['MOVEMENT2000K-g']}
    train_image_index = {'MOVEMENT1000K-b': [0, 799], 'MOVEMENT1000K-slab': [800, 1512], 'MOVEMENT2000K-l': [1513, 2312], 'MOVEMENT1300K-slab': [2313, 3112], 'MOVEMENT1500K-slab': [3113, 3912], 'MOVEMENT1000-300K-b': [3913, 4712], 'MOVEMENT2000K-g': [4713, 5512]}
    valid_image_index = {'MOVEMENT1000K-b': [0, 199], 'MOVEMENT1000K-slab': [200, 378], 'MOVEMENT2000K-l': [379, 578], 'MOVEMENT1300K-slab': [579, 778], 'MOVEMENT1500K-slab': [779, 978], 'MOVEMENT1000-300K-b': [979, 1178], 'MOVEMENT2000K-g': [1179, 1378]}
    kpu_column_name = ["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    rmse_column_name = ["batch","loss", "etot_lab", "etot_pre", "etot_rmse", "kpu_etot","ei_rmse", "f_lab", "f_pre", "f_rmse", "f_kpu", "kpu"]
    
    for root_dir in work_root_dir:
        temp_dict = {}
        for type in train_type:
            kpu_dict = {}
            rmse_dict = {}
            work_dir = WorkTrainDir(root_dir, type)
            kpu_dir = os.path.join(work_dir.log_dir, "kpu_dir")#kpu_dir_valid_10_image_p_1
            rmse_images = pd.read_csv(os.path.join(kpu_dir, "valid_kpu_force.csv"), index_col=0, header=0)
            for config in train_type:
                all_images_kpu = pd.DataFrame(columns=kpu_column_name)
                avg_images_kpu = pd.DataFrame(columns=rmse_column_name)
                for movement in config_dict[config]:
                    start = valid_image_index[movement][0]
                    end = valid_image_index[movement][1]#image_0.csv
                    for i in range(start, end+1):
                        file_name = "image_{}.csv".format(i)
                        force_kpu = pd.read_csv(os.path.join(kpu_dir, file_name), index_col=0, header=0, dtype=float)
                        force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                        force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                        all_images_kpu = pd.concat([all_images_kpu,force_kpu])
                        avg_kpu = list(rmse_images.loc[i])
                        avg_kpu.append(force_kpu["kpu"].mean())
                        avg_images_kpu.loc[i] = avg_kpu
                if avg_images_kpu.shape[0] > 200:
                    avg_images_kpu = avg_images_kpu.sample(200)
                kpu_dict[config] = all_images_kpu
                rmse_dict[config] = avg_images_kpu
                print("{} shape is {}".format(config, avg_images_kpu.shape))
                
            temp_dict[type] = rmse_dict
            picture_path = os.path.join(work_dir.work_dir, "{}-pictures-rmse-kpu-log".format(system))
            if os.path.exists(picture_path) is False:
                os.mkdir(picture_path)
            # etotal-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "etot_rmse", x_label = "KPU (etot)", y_label = "Etot rmse (eV)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_etot_rmse_kpu.png".format(type)))
            
            # draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "etot_pre", x_label = "KPU", y_label = "etot", title = "model trained with {} configurations".format(type), location = "best", \
            #             picture_save_path = os.path.join(picture_path, "{}_trained_etot_kpu.png".format(type)))
            
            # ei-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "ei_rmse", x_label = "KPU (etot)", y_label = "Ei rmse", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_Ei_rmse_kpu.png".format(type)))

            # force-kpu散点图f_pre
            draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "f_rmse", x_label = "KPU (force)", y_label = "Force rmse (eV/Å)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_force_rmse_kpu.png".format(type)))
            
            # draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "f_pre", x_label = "KPU", y_label = "force", title = "model trained with {} configurations".format(type), location = "best", \
            #             picture_save_path = os.path.join(picture_path, "{}_trained_force_kpu.png".format(type)))

"""
@Description :
横轴四个相名称，纵轴KPU, x_column="kpu" force KPU, "etot_kpu" energy kpu
@Returns     :
@Author       :wuxingxing
"""

def draw_cu_4pahses_kpu_pashe():
    def draw_scatter(x_list:list, kpu_list :list, legend_label:list, x_column, y_column, x_label, y_label, title, location, picture_save_path, draw_config = None):
        # force-kpu散点图
        font =  {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 20,
            }
        plt.figure(figsize=(12,9))
        plt.style.use('classic') # 画板主题风格
        plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
        plt.grid() # 网格线
        start = None
        end = None
        for i in range(len(kpu_dict)):
            x_range = [j for j in range(start, end)]#plt.yscale('log')
            plt.scatter(x_list[i], kpu_list[i], \
                color=color_list[i], marker=mark_list[i], \
                    label=legend_label[i])
        
        plt.xticks([100,300,500,700], ["bulk", "gas", "slab", "liquid"], fontsize=25)
        # plt.xlabel(x_label, font)
        plt.yscale('log')
        plt.ylabel(y_label, font)
        plt.title(title, font)
        plt.legend(fontsize=12, frameon=False, loc=location)
        plt.savefig(picture_save_path)

    #设置工作路径
    work_root_dir = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/cu_4pahses"  #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/share/home/wuxingxing/datas/system_config/cu_72104/dft_test/dft_cu_4phases/bulk_gas_slab_liquid.csv"
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    for root_dir in work_root_dir:
        temp_dict = {}
        for type in train_type:
            kpu_dict = {}
            rmse_dict = {}
            work_dir = WorkTrainDir(root_dir, type)
            kpu_dir = os.path.join(work_dir.log_dir, "kpu_dir")#kpu_dir_valid_10_image_p_1
            rmse_images = pd.read_csv(os.path.join(kpu_dir, "valid_kpu_force.csv"), index_col=0, header=0)
            for config in train_type:
                # all_images_kpu = pd.DataFrame(columns=kpu_column_name)
                avg_images_kpu = pd.DataFrame(columns=rmse_column_name)
                for movement in config_dict[config]:
                    start = valid_image_index[movement][0]
                    end = valid_image_index[movement][1]#image_0.csv
                    for i in range(start, end+1):
                        file_name = "image_{}.csv".format(i)
                        force_kpu = pd.read_csv(os.path.join(kpu_dir, file_name), index_col=0, header=0, dtype=float)
                        force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                        force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                        # 随机选择200数据
                        # all_images_kpu = pd.concat([all_images_kpu,force_kpu])
                        avg_kpu = list(rmse_images.loc[i])
                        avg_kpu.append(force_kpu["kpu"].mean())
                        avg_images_kpu.loc[i] = avg_kpu
                if avg_images_kpu.shape[0] > 200:
                    avg_images_kpu = avg_images_kpu.sample(200)

                # kpu_dict[config] = all_images_kpu
                rmse_dict[config] = avg_images_kpu
                print("{} shape is {}".format(config, avg_images_kpu.shape))
            temp_dict[type] = rmse_dict
            picture_path = os.path.join(work_dir.work_dir, "{}-pictures-phase-kpu-log(kpu)".format(system))
            if os.path.exists(picture_path) is False:
                os.mkdir(picture_path)
                    # etotal-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "etot_rmse", \
                x_label = "phases", y_label = "KPU (etot)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_etot_kpu_phases.png".format(type)), draw_config = type)
            draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "f_rmse", \
                x_label = "phases", y_label = "KPU (force)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_force_kpu_phases.png".format(type)), draw_config = type)

"""
@Description :
横轴 epoch 纵轴KPU 线条 随着epoch增加 kpu变化趋势
@Returns     :
@Author       :wuxingxing
"""

def draw_cu_4pahses_epoch_kpu():
    def draw_lines(train_type, kpu_dict, x_column, y_column, x_label, y_label, title, location, picture_save_path):
        # force-kpu散点图
        font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size' : 20,
        }
        plt.figure(figsize=(12,9))
        plt.style.use('classic') # 画板主题风格
        plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
        plt.grid() # 网格线
        for i in range(len(train_type)):
            # kpu_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][x_column].min(),4),round(kpu_dict[train_type[i]][x_column].max(),4))
            # y_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][y_column].min(),4),round(kpu_dict[train_type[i]][y_column].max(),4))
            # legend_label = "{} with {} images, KPU values {}, {} values {}".format(train_type[i], kpu_dict[train_type[i]].shape[0], kpu_range, y_label, y_range)
            # y_kpu = [math.log10(k) for k in (list(kpu_dict[train_type[i]][x_column]))]
            # print(title, ", ", legend_label)
            plt.plot(kpu_dict[train_type[i]][x_column], kpu_dict[train_type[i]][y_column], \
                color=color_list[i], marker=mark_list[i], \
                    label=train_type[i], linewidth=3)
        plt.yscale('log')
        plt.xlabel(x_label, font)
        plt.ylabel(y_label, font)
        plt.title(title, font)
        plt.legend(fontsize=16, frameon=False, loc=location)
        # plt.show()
        plt.savefig(picture_save_path)
            
    #设置工作路径
    system = "cu_4phases" # cu_4phase #cu_bulk #cuo
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system".format(system)]   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/data/data/wuxingxing/datas/{}/train_data".format(system)
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    train_type = ["bulk", "slab"]# , "liquid", "gas"
    config_dict = {"bulk":['MOVEMENT1000K-b','MOVEMENT1000-300K-b'], "slab":['MOVEMENT1000K-slab', 'MOVEMENT1300K-slab', 'MOVEMENT1500K-slab'], "liquid":['MOVEMENT2000K-l'], "gas":['MOVEMENT2000K-g']}
    train_image_index = {'MOVEMENT1000K-b': [0, 799], 'MOVEMENT1000K-slab': [800, 1512], 'MOVEMENT2000K-l': [1513, 2312], 'MOVEMENT1300K-slab': [2313, 3112], 'MOVEMENT1500K-slab': [3113, 3912], 'MOVEMENT1000-300K-b': [3913, 4712], 'MOVEMENT2000K-g': [4713, 5512]}
    valid_image_index = {'MOVEMENT1000K-b': [0, 199], 'MOVEMENT1000K-slab': [200, 378], 'MOVEMENT2000K-l': [379, 578], 'MOVEMENT1300K-slab': [579, 778], 'MOVEMENT1500K-slab': [779, 978], 'MOVEMENT1000-300K-b': [979, 1178], 'MOVEMENT2000K-g': [1179, 1378]}
    kpu_column_name = ["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    rmse_column_name = ["batch","loss", "etot_lab", "etot_pre", "etot_rmse", "kpu_etot","ei_rmse", "f_lab", "f_pre", "f_rmse", "f_kpu", "kpu"]
    epoch_avg_column_name = ["epoch", "etot_kpu", "force_kpu"]
    image_dict = {"bulk":[10, 11, 12, 13, 14, 1000, 1001, 1002, 1003, 1004],
                "slab":[200, 201, 202, 203, 204, 580, 581, 582, 583, 584],
                "liquid":[380, 381, 382, 383, 384, 385, 386, 387, 388, 389],
                "gas": [1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209]}
    for root_dir in work_root_dir:
        temp_dict = {}
        for type in train_type:
            kpu_dict = {}
            rmse_dict = {}
            epoch_avg_dict = {}
            work_dir = WorkTrainDir(root_dir, type)
            for config in train_type:
                epoch_avg_df = pd.DataFrame(columns=epoch_avg_column_name)
                for epoch in range(5, 301, 5):
                    rmse_images = pd.read_csv(os.path.join(work_dir.log_dir, "valid_10image_kpu_epoch_0_300/kpu_dir_epoch_{}/{}".format(epoch, "valid_kpu_force.csv")), index_col=0, header=0)
                    # all_images_kpu = pd.DataFrame(columns=kpu_column_name)
                    avg_images_kpu = pd.DataFrame(columns=rmse_column_name)
                    for i in image_dict[config]:
                        file_name = "image_{}.csv".format(i)
                        force_kpu = pd.read_csv(os.path.join(work_dir.log_dir, "valid_10image_kpu_epoch_0_300/kpu_dir_epoch_{}/{}".format(epoch, file_name)), index_col=0, header=0, dtype=float)
                        force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                        force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                        avg_kpu = list(rmse_images.loc[i])
                        avg_kpu.append(force_kpu["kpu"].mean())
                        avg_images_kpu.loc[i] = avg_kpu
                    epoch_avg_df.loc[epoch]=[epoch, avg_images_kpu["kpu_etot"].mean(), avg_images_kpu["kpu"].mean()]
                epoch_avg_dict[config] = epoch_avg_df
                # kpu_dict[config] = all_images_kpu
                # rmse_dict[config] = avg_images_kpu
                print("{} shape is {}, epoch_avg_df:".format(config, epoch_avg_df.shape))
                print(epoch_avg_df)
            picture_path = os.path.join(work_dir.work_dir, "{}-pictures-epoch-etot-kpu-log(kpu)".format(system))
            if os.path.exists(picture_path) is False:
                os.mkdir(picture_path)
            draw_lines(train_type, epoch_avg_dict, x_column = "epoch", y_column = "etot_kpu", \
                x_label = "epochs", y_label = "KPU (etot)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_epoch_300_etot_kpu.png".format(type)))
            draw_lines(train_type, epoch_avg_dict, x_column = "epoch", y_column = "force_kpu", \
                x_label = "epochs", y_label = "KPU (force)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_epoch_300_force_kpu.png".format(type)))

def draw_li_kpu_pashe():
    def draw_scatter(train_type, kpu_dict, x_column, y_column, x_label, y_label, title, location, picture_save_path, draw_config = None):
        # force-kpu散点图
        font =  {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 20,
            }
        plt.figure(figsize=(12,9))
        plt.style.use('classic') # 画板主题风格
        plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
        plt.grid() # 网格线
        start = None
        end = None
        for i in range(len(train_type)):
            kpu_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][x_column].min(),4),round(kpu_dict[train_type[i]][x_column].max(),4))
            y_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][y_column].min(),4),round(kpu_dict[train_type[i]][y_column].max(),4))
            legend_label = "{} with {} images, KPU {}".format(train_type[i], kpu_dict[train_type[i]].shape[0], kpu_range)
            start = 0 if start is None else end
            end = kpu_dict[train_type[i]].shape[0] if end is None else kpu_dict[train_type[i]].shape[0] + end
            x_range = [j for j in range(start, end)]
            # y_kpu = [math.log10(k) for k in (list(kpu_dict[train_type[i]][x_column]))]
            print("{} is {}, x_s:{}, x_e:{}, len({})".format(title, legend_label, start, end, len(x_range)))
            plt.scatter(x_range, kpu_dict[train_type[i]][x_column], \
                color=color_list[i], marker=mark_list[i], \
                    label=legend_label)
        
        # plt.xlabel(x_label, font)
        plt.xticks([200,600,1000], [" ", " ", " "])
        plt.yscale('log')
        plt.ylabel(y_label)
        plt.title(title, font)
        plt.legend(fontsize=12, frameon=False, loc=location)
        # plt.show()
        plt.savefig(picture_save_path)

    #设置工作路径
    system = "li" # cu_4phase #cu_bulk #cuo
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system".format(system)]   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/data/data/wuxingxing/datas/{}/train_data".format(system)
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    train_type = ["bcc", "fcc", "hcp"]
    config_dict = {"bcc":['bcc/10','bcc/8','bcc/7','bcc/6','bcc/9','bcc/3','bcc/5','bcc/1','bcc/4','bcc/2'],\
                    "fcc":['fcc/10','fcc/8','fcc/7','fcc/6','fcc/9','fcc/3','fcc/5','fcc/1','fcc/4','fcc/2'], \
                        "hcp":['hcp/10','hcp/8','hcp/7','hcp/6','hcp/9','hcp/3','hcp/5','hcp/1','hcp/4','hcp/2']}
    train_image_index = {'bcc': [0, 1599], 'fcc': [1600, 3199], 'hcp': [3200, 4799]}
    valid_image_index = {'bcc': [0, 399], 'fcc': [400, 799], 'hcp': [800, 1199]}

    #设置工作路径
    kpu_column_name = ["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    rmse_column_name = ["batch","loss", "etot_lab", "etot_pre", "etot_rmse", "kpu_etot","ei_rmse", "f_lab", "f_pre", "f_rmse", "f_kpu", "kpu"]
    for root_dir in work_root_dir:
        temp_dict = {}
        for type in train_type:
            kpu_dict = {}
            rmse_dict = {}
            work_dir = WorkTrainDir(root_dir, type)
            rmse_images = pd.read_csv(os.path.join(work_dir.log_dir, "kpu_dir/{}".format("valid_kpu_force.csv")), index_col=0, header=0)
            for config in train_type:
                all_images_kpu = pd.DataFrame(columns=kpu_column_name)
                avg_images_kpu = pd.DataFrame(columns=rmse_column_name)
                start = valid_image_index[config][0]
                end = valid_image_index[config][1]#image_0.csv
                for i in range(start, end+1):
                    file_name = "image_{}.csv".format(i)
                    force_kpu = pd.read_csv(os.path.join(work_dir.log_dir, "kpu_dir/{}".format(file_name)), index_col=0, header=0, dtype=float)
                    force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                    force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                    all_images_kpu = pd.concat([all_images_kpu,force_kpu])
                    avg_kpu = list(rmse_images.loc[i])
                    avg_kpu.append(force_kpu["kpu"].mean())
                    avg_images_kpu.loc[i] = avg_kpu
                kpu_dict[config] = all_images_kpu
                rmse_dict[config] = avg_images_kpu
                print("{} shape is {}".format(config, avg_images_kpu.shape))
            temp_dict[type] = rmse_dict
            picture_path = os.path.join(work_dir.work_dir, "{}-pictures-kpu_etot-log(kpu)".format(system))
            if os.path.exists(picture_path) is False:
                os.mkdir(picture_path)
                    # etotal-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "etot_rmse", x_label = "phases", y_label = "KPU", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_kpu_phases.png".format(type)), draw_config = type)

def draw_li_rmse_kpu():
    def draw_scatter(train_type, kpu_dict, x_column, y_column, x_label, y_label, title, location, picture_save_path):
            # force-kpu散点图
            font =  {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 20,
            }
            plt.figure(figsize=(12,9))
            plt.style.use('classic') # 画板主题风格
            plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
            plt.grid() # 网格线
            for i in range(len(train_type)):
                kpu_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][x_column].min(),4),round(kpu_dict[train_type[i]][x_column].max(),4))
                y_range = "[{}-{}]".format(round(kpu_dict[train_type[i]][y_column].min(),4),round(kpu_dict[train_type[i]][y_column].max(),4))
                legend_label = "{} with {} images, KPU values {}, {} values {}".format(train_type[i], kpu_dict[train_type[i]].shape[0], kpu_range, y_label, y_range)
                y_kpu = [math.log10(k) for k in (list(kpu_dict[train_type[i]][x_column]))]
                print(title, ", ", legend_label)
                plt.scatter(y_kpu, kpu_dict[train_type[i]][y_column], \
                    color=color_list[i], marker=mark_list[i], \
                        label=legend_label)
            plt.xlabel(x_label, font)
            plt.ylabel(y_label, font)
            plt.title(title, font)
            plt.legend(fontsize=12, frameon=False, loc=location)
            # plt.show()
            plt.savefig(picture_save_path)

    #设置工作路径
    #设置工作路径
    system = "li" # cu_4phase #cu_bulk #cuo
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system".format(system)]   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
    data_path = "/data/data/wuxingxing/datas/{}/train_data".format(system)
    # train_image_index, valid_image_index = get_image_index(system) # this function in active_learning.util script.
    train_type = ["bcc", "fcc", "hcp"]
    config_dict = {"bcc":['bcc/10','bcc/8','bcc/7','bcc/6','bcc/9','bcc/3','bcc/5','bcc/1','bcc/4','bcc/2'],\
                    "fcc":['fcc/10','fcc/8','fcc/7','fcc/6','fcc/9','fcc/3','fcc/5','fcc/1','fcc/4','fcc/2'], \
                        "hcp":['hcp/10','hcp/8','hcp/7','hcp/6','hcp/9','hcp/3','hcp/5','hcp/1','hcp/4','hcp/2']}
    train_image_index = {'bcc': [0, 1599], 'fcc': [1600, 3199], 'hcp': [3200, 4799]}
    valid_image_index = {'bcc': [0, 399], 'fcc': [400, 799], 'hcp': [800, 1199]}
    
    kpu_column_name = ["atom_index", "loss_x", "loss_y", "loss_z", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre", "loss", "kpu"]
    rmse_column_name = ["batch","loss", "etot_lab", "etot_pre", "etot_rmse", "kpu_etot","ei_rmse", "f_lab", "f_pre", "f_rmse", "f_kpu", "kpu"]
    for root_dir in work_root_dir:
        temp_dict = {}
        for type in train_type:
            kpu_dict = {}
            rmse_dict = {}
            work_dir = WorkTrainDir(root_dir, type)
            rmse_images = pd.read_csv(os.path.join(work_dir.log_dir, "kpu_dir/{}".format("valid_kpu_force.csv")), index_col=0, header=0)
            for config in train_type:
                all_images_kpu = pd.DataFrame(columns=kpu_column_name)
                avg_images_kpu = pd.DataFrame(columns=rmse_column_name)
                start = valid_image_index[config][0]
                end = valid_image_index[config][1]#image_0.csv
                for i in range(start, end+1):
                    file_name = "image_{}.csv".format(i)
                    force_kpu = pd.read_csv(os.path.join(work_dir.log_dir, "kpu_dir/{}".format(file_name)), index_col=0, header=0, dtype=float)
                    force_kpu["loss"] = (abs(force_kpu["loss_x"]) + abs(force_kpu["loss_y"]) + abs(force_kpu["loss_z"]))/3
                    force_kpu["kpu"] = (abs(force_kpu["kpu_x"]) + abs(force_kpu["kpu_y"]) + abs(force_kpu["kpu_z"]))/3
                    all_images_kpu = pd.concat([all_images_kpu,force_kpu])
                    avg_kpu = list(rmse_images.loc[i])
                    avg_kpu.append(force_kpu["kpu"].mean())
                    avg_images_kpu.loc[i] = avg_kpu
                kpu_dict[config] = all_images_kpu
                rmse_dict[config] = avg_images_kpu
                print("{} shape is {}".format(config, avg_images_kpu.shape))
            temp_dict[type] = rmse_dict
            picture_path = os.path.join(work_dir.work_dir, "{}-pictures-kpu_etot-log".format(system))
            if os.path.exists(picture_path) is False:
                os.mkdir(picture_path)
            # etotal-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "etot_rmse", x_label = "KPU", y_label = "Etot rmse (eV)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_etot_rmse_kpu.png".format(type)))
            
            # draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "etot_pre", x_label = "KPU", y_label = "etot", title = "model trained with {} configurations".format(type), location = "best", \
            #             picture_save_path = os.path.join(picture_path, "{}_trained_etot_kpu.png".format(type)))
            
            # ei-rmse-kpu散点图
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "ei_rmse", x_label = "KPU", y_label = "Ei rmse", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_Ei_rmse_kpu.png".format(type)))

            # force-kpu散点图f_pre
            draw_scatter(train_type, rmse_dict, x_column = "kpu_etot", y_column = "f_rmse", x_label = "KPU", y_label = "Force rmse (eV/Å)", title = "model trained with {} configurations".format(type), location = "best", \
                        picture_save_path = os.path.join(picture_path, "{}_trained_force_rmse_kpu.png".format(type)))
            
            # draw_scatter(train_type, rmse_dict, x_column = "kpu", y_column = "f_pre", x_label = "KPU", y_label = "force", title = "model trained with {} configurations".format(type), location = "best", \
            #             picture_save_path = os.path.join(picture_path, "{}_trained_force_kpu.png".format(type)))
        
if __name__ == "__main__":
    draw_H()
    # draw_kpu()
    # draw_cu_4pahses_rmse_kpu()
    draw_cu_4pahses_kpu_pashe()
    # draw_cu_4pahses_epoch_kpu()
    # draw_li_kpu_pashe()
    # draw_li_rmse_kpu()
