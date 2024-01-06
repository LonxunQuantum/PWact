from ftplib import error_perm
from mailbox import linesep
from operator import index
import os
from turtle import color
import numpy as np
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pyparsing import col
from torch import rand
from draw_pictures.workdir import WorkTrainDir
import active_learning.active_learning_params as al_pm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import linregress

color_list = ["#000000", "#008B8B", "#BDB76B", "#B8860B", "#FF8C00", "#A52A2A"]
mark_list = [".", ",", "v", "^", "+"]
def draw_model_MSE():
    model_num = 5
    #"batch","m_0","Loss_0","E_tot_0","Ei_0","E_f_0","HPHt_0",\
    # "m_1","Loss_1","E_tot_1","Ei_1","E_f_1","HPHt_1",\
    #   ...
    # "m_4","Loss_4","E_tot_4","Ei_4","E_f_4","HPHt_4"
    # df = pd.read(al_pm.valid_loss_drop_path, index_col=0, header=0, dtype=float)
    df = pd.read_csv(al_pm.valid_loss_drop_path, index_col=0, header=0, dtype=float)
    for i in range(model_num):
        df['E_tot_{}'.format(i)] = df['E_tot_{}'.format(i)]*108
    df = df.sample(frac=0.3, axis=0)
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    x = range(0, df.index.values.__len__())
    for i in range(model_num):
        plt.plot(x, df['E_tot_{}'.format(i)], linewidth=1, color=color_list[i], marker=mark_list[i],label="Etotal MSE of model_{}".format(i))

    plt.xlabel("images")
    plt.ylabel("E_total MSE")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(al_pm.valid_dir, "Etotal_mse.png"))

    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    x = range(0, df.index.values.__len__())
    for i in range(model_num):
        plt.plot(x, df['E_f_{}'.format(i)], linewidth=1, color=color_list[i], marker=mark_list[i],label="Etotal MSE of model_{}".format(i))
    plt.xlabel("images")
    plt.ylabel("atom_force MSE")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(al_pm.valid_dir, "atom_force.png"))   #al_pm.valid_dir


def draw_unc_MSE():
    model_num = 5
    column_E_t = []
    column_E_i = []
    column_F = []
    for i in range(model_num):
        column_E_t.append("E_tot_{}".format(i))
        column_E_i.append("Ei_{}".format(i))
        column_F.append("E_f_{}".format(i))
    #"batch","m_0","Loss_0","E_tot_0","Ei_0","E_f_0","HPHt_0",\
    # "m_1","Loss_1","E_tot_1","Ei_1","E_f_1","HPHt_1",\
    #   ...
    # "m_4","Loss_4","E_tot_4","Ei_4","E_f_4","HPHt_4"
    # df = pd.read(al_pm.valid_loss_drop_path, index_col=0, header=0, dtype=float)
    import random
    index_num = 50
    rows = random.sample(range(0,330), index_num)

    all_path = ["init_5_al_ekf_dir", "init_10_al_ekf_dir", "init_20_al_ekf_dir", "init_best_al_ekf_dir"]
    labels = ['5', '10', '20', '100']
    path = "/mlff_wu_work_dir/al_cu_bulk_drop_dir/init_20_al_ekf_dir/act_lea_log_dir/drop_valid_log"
    df_dict = {}
    for i in all_path:
        path_i = './mlff_wu_work_dir/al_cu_bulk_drop_dir/{}/act_lea_log_dir/drop_valid_log/valid_loss_nodrop.csv'.format(i)
        df = pd.read_csv(path_i, index_col=0, header=0, dtype=float)
        df = df.loc[rows]
        df[column_E_t] = df[column_E_t] / 108   #etotal / atom nums
        df['mean_E_t'] = (df[column_E_t]).mean(axis=1)
        df['mean_E_i'] = df[column_E_i].mean(axis=1)
        df['mean_F'] = df[column_F].mean(axis=1)
        for j in range(model_num):
            df[column_E_t[j]] = (df[column_E_t[j]] - df['mean_E_t'])**2
            df[column_E_i[j]] = (df[column_E_i[j]] - df['mean_E_i'])**2
            df[column_F[j]] = (df[column_F[j]] - df['mean_F'])**2
        df['mse_E_t'] = df[column_E_t].mean(axis=1)
        df['mse_E_i'] = df[column_E_i].mean(axis=1)
        df['mse_F'] = df[column_F].mean(axis=1)
        print(df[['mse_E_t','mse_E_i','mse_F']])
        df_dict[i] = df

    #draw E_t
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    x = range(0, len(rows))
    for i in range(len(all_path)):
        plt.plot(x, df_dict[all_path[i]]['mse_E_t'], linewidth=1, color=color_list[i], marker=mark_list[i],label="MSE of model with {} training set".format(labels[i]))
    plt.xlabel("images")
    plt.ylabel("E_total MSE")
    plt.legend(loc='best')
    plt.title("Variation of models in different scale data".format(model_num))
    # plt.show()
    plt.savefig(os.path.join(al_pm.al_work_dir, "Variation of {} models in E_total".format(model_num)))

    #draw E_i
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    x = range(0, len(rows))
    for i in range(len(all_path)):
        plt.plot(x, df_dict[all_path[i]]['mse_E_i'], linewidth=1, color=color_list[i], marker=mark_list[i],label="MSE of model with {} training set".format(labels[i]))
    plt.xlabel("images")
    plt.ylabel("E_i MSE")
    plt.legend(loc='best')
    plt.title("Variation of models in different scale data".format(model_num))
    # plt.show()
    plt.savefig(os.path.join(al_pm.al_work_dir, "Variation of {} models in E_i".format(model_num)))

    #draw E_F
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    x = range(0, len(rows))
    for i in range(len(all_path)):
        plt.plot(x, df_dict[all_path[i]]['mse_F'], linewidth=1, color=color_list[i], marker=mark_list[i],label="MSE of model with {} training set".format(labels[i]))
    plt.xlabel("images")
    plt.ylabel("Force MSE")
    plt.legend(loc='best')
    plt.title("Variation of models in different scale data".format(model_num))
    # plt.show()
    plt.savefig(os.path.join(al_pm.al_work_dir, "Variation of {} models in Force".format(model_num)))

def read_drop_variance_loss():
    loss_dict = {}
    rmse_dict = {}
    variation_dict = {}
    model_num = 5
    column_E_t = []
    column_E_i = []
    column_F = []
    for i in range(model_num):
        column_E_t.append("e_tot_rmse{}".format(i))
        column_E_i.append("ei_rmse{}".format(i))
        column_F.append("e_f_rmse{}".format(i))

    system = "cu_4phases_system" # cu_bulk_system cu_4phase_system li_system cuo cuc
    data_path = "/data/data/wuxingxing/datas/{}/train_data".format(system)
    work_root_dir = ["mlff_wu_work_dir/dpnn_work_dir/{}_system_drop20".format(system)]
    train_type = ["bulk", "slab", "liquid", "gas"]
    valid_type = [5, 10, 20]
    for root_dir in work_root_dir:
        for t_type in train_type:
            work_dir = WorkTrainDir(root_dir, t_type)
            work_dir.set_torch_data_path(data_path)
            mse_dir = os.path.join(work_dir.log_dir, "mse_dir")
            valid_dict_df = {}
            valid_dict_etot_mse = {}
            etot_mse_dict = {}
            for drop in valid_type:
                etot_mse = pd.read_csv(os.path.join(mse_dir, "etot_mse_{}.csv".format(drop)), index_col=0, header=0) #['batch', 'atom_num', 'e_tot_mse', 'e_tot_mse_atom', 'ei_variance', 'f_variance']
                df = pd.read_csv(os.path.join(work_dir.log_dir,"valid_loss_nodrop_{}.csv".format(drop)), index_col=0, header=0)
                df['mean_E_t'] = (df[column_E_t]).mean(axis=1)
                df['mean_E_i'] = df[column_E_i].mean(axis=1)
                df['mean_F'] = df[column_F].mean(axis=1)
                print(df[['mean_E_t','mean_E_i','mean_F']].shape)
                valid_dict_df[drop] = df
                valid_dict_etot_mse[drop] = valid_dict_etot_mse
                etot_mse_dict[drop] = etot_mse
            loss_dict[t_type] = valid_dict_df
            variation_dict[t_type] = valid_dict_etot_mse
            rmse_dict[t_type] = etot_mse_dict

    return work_dir, loss_dict, variation_dict

def draw_variation():
    train_data_percent = [5, 10, 20, 100]
    work_dir, loss_dict, variation_dict, rmse_dict = read_drop_variance_loss()

    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线

    j = 0
    for i in loss_dict.keys():
        x = variation_dict[i].index.values
        plt.plot(x, variation_dict[i]["f_variance"], linewidth=1, color=color_list[j], marker=mark_list[j],label="Force variation of models with {} training set".format(train_data_percent[j]))
        j = j + 1

    # ax.set_xticks(x)
    plt.xlim(0, max(x))
    plt.xlabel("images")
    plt.ylabel("Force variation (mse)")
    plt.legend(loc='best')
    plt.title("Force variation of models in different scale data")
    # plt.show()
    plt.savefig(os.path.join(work_dir.log_dir, "force variation of models.png"))

def draw_variation_scatter():
    train_data_percent = [10, 20, 100]# 100
    work_dir, loss_dict, variation_dict, rmse_dict = read_drop_variance_loss()

    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线

    # 力散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    j = 0
    for i in loss_dict.keys():
        plt.scatter(variation_dict[i]["f_variance"], loss_dict[i]["mean_F"], color=color_list[j], marker=mark_list[j], label="Force variation of models with {}% training set".format(train_data_percent[j]))
        j = j+1
    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    plt.ylabel("forece loss")
    plt.xlabel("Force variance (mse)")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(work_dir.type_dir, "force scatter of models.png"))

    #Ei散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    plt.grid() # 网格线
    j = 0
    for i in loss_dict.keys():
        plt.scatter(variation_dict[i]["ei_variance"], loss_dict[i]["mean_E_i"], color=color_list[j], marker=mark_list[j], label="Ei variation of models with {}% training set".format(train_data_percent[j]))
        j = j+1
    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    plt.ylabel("Ei loss")
    plt.xlabel("Ei variance (mse)")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(work_dir.type_dir, "Ei scatter of models.png"))

    #E_total散点图
    plt.figure(figsize=(18,12))
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体

    plt.grid() # 网格线
    j = 0
    for i in loss_dict.keys():
        #"e_tot_mse", "e_tot_mse_atom" variance of e-total
        plt.scatter(variation_dict[i]["e_tot_mse_atom"], loss_dict[i]["mean_E_t"], color=color_list[j], marker=mark_list[j], label="E total variation of models with {}% training set".format(train_data_percent[j]))
        j = j+1
    # plt.title(r"reight images {} with pre_etotoal in {}; cadidate images {} with pre_etotoal in {}; error images:{}".format(len(reight_pre), 0.06,len(cadidate_pre), 0.1, len(error_pre)))
    
    plt.ylabel("E_total loss")
    plt.xlabel("E_total variance (mse)")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(work_dir.type_dir, "E_total scatter of models.png"))

if __name__=="__main__":
    # draw_model_MSE()
    # draw_unc_MSE()
    draw_variation()
    # draw_variation_scatter()