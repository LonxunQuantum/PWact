from email.utils import getaddresses
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.file_operation import del_file
import active_learning.active_learning_params as al_pm
picture_save_dir = "./mlff_wu_work_dir/picture_dir"

def read_data_as_dataframe(file_path):
    df = pd.DataFrame(columns=al_pm.retrain_train_iter_loss_cloumns)
    i = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            df.loc[i] = [float(j) for j in line.split()]
            i = i + 1
    print("{} shape is {}".format(file_path, df.shape))
    return df

def get_data(dir):
    train_epoch_loss = read_data_as_dataframe(os.path.join(dir), "retrain_train_epoch_loss.dat")
    train_iter_loss = read_data_as_dataframe(os.path.join(dir), "retrain_train_iter_loss.dat")
    valid_epoch_loss = read_data_as_dataframe(os.path.join(dir), "retrain_valid_epoch_loss.dat")
    valid_iter_loss = read_data_as_dataframe(os.path.join(dir), "retrain_valid_iter_loss.dat")
    return train_epoch_loss, train_iter_loss, valid_epoch_loss, valid_iter_loss

def get_epoch_loss(file_path):
    epoch_loss = None
    i = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "epoch" in line:
                epoch_loss = pd.DataFrame(columns=[j for j in line.split()])
            else:
                epoch_loss.loc[i] = [float(j) for j in line.split()]
            i = i + 1
    return epoch_loss

def get_epoch_loss_valid(file_path):
    epoch_loss_valid = None
    i = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "epoch" in line:
                epoch_loss_valid = pd.DataFrame(columns=[j for j in line.split()])
            else:
                epoch_loss_valid.loc[i] = [float(j) for j in line.split()]
            i = i + 1
    return epoch_loss_valid

def get_iter_loss(file_path):
    i = 0
    column_name = ['iter', 'loss', 'RMSE_Etot', 'RMSE_Ei', 'RMSE_F', 'lr', 'time(s)']
    iter_loss = pd.DataFrame(columns=column_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            iter_loss.loc[i] = [float(j) for j in line.split()]
            i = i + 1

    return iter_loss

def get_loss_valid(file_path):
    i = 0
    column_name = ["iter", "loss", "RMSE_Etot", "RMSE_Ei", "RMSE_F", "lr"]
    loss_valid = pd.DataFrame(columns=column_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            loss_valid.loc[i] = [float(j) for j in line.split()]
            i = i + 1

    return loss_valid

def get_etot(dir_path):
    dict = {}
    epoch_loss_path = os.path.join(dir_path, "retrain_train_epoch_loss.dat")
    epoch_loss = get_epoch_loss(epoch_loss_path)

    epoch_loss_valid = os.path.join(dir_path, "retrain_valid_epoch_loss.dat")
    epoch_loss_valid = get_epoch_loss_valid(epoch_loss_valid)

    iter_loss = os.path.join(dir_path, "retrain_train_iter_loss.dat")
    iter_loss = get_iter_loss(iter_loss)
    #
    loss_valid = os.path.join(dir_path, "retrain_valid_iter_loss.dat")
    loss_valid = get_loss_valid(loss_valid)

    dict['epoch_loss_train'] = epoch_loss
    dict['epoch_loss_valid'] = epoch_loss_valid
    dict['iter_train_loss'] = iter_loss
    dict['iter_valid_loss'] = loss_valid

    return dict

def draw_epoch_loss_train(p_dicts):
    fig_1 = plt.figure(1, figsize=(13, 4))
    cnt = 1
    draw_types = ['loss', 'RMSE_Etot', 'RMSE_F']

    labels = []
    for type in draw_types:
        ax = plt.subplot(1, 3, cnt)
        ax.set_title("train_epoch_" + type)
        # ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        for key in p_dicts.keys():
            if draw_types.index(type) == 0:
                ax.plot(p_dicts[key]['epoch_loss_train'][type].values, label="%s_%s" % (key, type))
            else:
                 ax.plot(p_dicts[key]['epoch_loss_train'][type].values)
        cnt += 1
        ax.grid()
        ax.legend()
    # handles, labels= ax.get_legend_handles_labels()
    # fig_1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
    #              borderaxespad=0, bbox_transform=fig_1.transFigure, ncol=2)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(picture_save_dir, "train_epoch.png"))

def draw_epoch_loss_valid(p_dicts):
    fig_1 = plt.figure(1, figsize=(13, 4))
    cnt = 1
    draw_types = ['valid_RMSE_Etot', 'valid_RMSE_F']

    labels = []
    for type in draw_types:
        ax = plt.subplot(1, 2, cnt)
        ax.set_title("valid_epoch_" + type)
        # ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        for key in p_dicts.keys():
            if draw_types.index(type) == 0:
                ax.plot(p_dicts[key]['epoch_loss_valid'][type].values, label="%s_%s" % (key, type))
            else:
                 ax.plot(p_dicts[key]['epoch_loss_valid'][type].values)
        cnt += 1
        ax.grid()
        ax.legend()
    # handles, labels= ax.get_legend_handles_labels()
    # fig_1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
    #              borderaxespad=0, bbox_transform=fig_1.transFigure, ncol=2)
    # plt.tight_layout()
    plt.savefig(os.path.join(picture_save_dir, "valid_epoch.png"))

def draw_iter_train_loss(p_dicts):
    fig_1 = plt.figure(1, figsize=(13, 4))
    cnt = 1
    # ["iter", "loss", "RMSE_Etot", "RMSE_Ei", "RMSE_F", "lr", "time"]
    draw_types = ["loss", 'RMSE_Etot', 'RMSE_F']

    labels = []
    for type in draw_types:
        ax = plt.subplot(1, 3, cnt)
        ax.set_title("iter_train_loss_" + type)
        # ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        for key in p_dicts.keys():
            if draw_types.index(type) == 0:
                ax.plot(p_dicts[key]['iter_train_loss'][type].values, label="%s_%s" % (key, type))
            else:
                 ax.plot(p_dicts[key]['iter_train_loss'][type].values)
        cnt += 1
        ax.grid()
        ax.legend()
    # handles, labels= ax.get_legend_handles_labels()
    # fig_1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
    #              borderaxespad=0, bbox_transform=fig_1.transFigure, ncol=2)
    # plt.tight_layout()
    plt.savefig(os.path.join(picture_save_dir, "iter_train_loss.png"))


def draw_iter_valid_loss(p_dicts):
    fig_1 = plt.figure(1, figsize=(13, 4))
    cnt = 1
    # "iter", "loss", "RMSE_Etot", "RMSE_Ei", "RMSE_F", "lr"
    draw_types = ["loss", 'RMSE_Etot', 'RMSE_F']

    labels = []
    for type in draw_types:
        ax = plt.subplot(1, 3, cnt)
        ax.set_title("iter_valid_loss_" + type)
        # ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        for key in p_dicts.keys():
            if draw_types.index(type) == 0:
                ax.plot(p_dicts[key]['iter_valid_loss'][type].values, label="%s_%s" % (key, type))
            else:
                 ax.plot(p_dicts[key]['iter_valid_loss'][type].values)
        cnt += 1
        ax.grid()
        ax.legend()
    # handles, labels= ax.get_legend_handles_labels()
    # fig_1.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.35, -0.01),
    #              borderaxespad=0, bbox_transform=fig_1.transFigure, ncol=2)
    # plt.tight_layout()
    plt.savefig(os.path.join(picture_save_dir, "iter_valid_loss.png"))

def draw_retrain_epoch_iter_train_valid():
    dir = al_pm.opt_log_dir
    # [ "retrain_num", "epoch", "iter", "loss", "RMSE_Etot", "RMSE_Ei", "RMSE_F", "lr", "time"]
    train_epoch_loss, train_iter_loss, valid_epoch_loss, valid_iter_loss = get_data(dir)
    x_labels = ["train_epoch", "train_iter", "valid_epoch", "valid_iter"]
    titles = ["train_epoch", "train_iter", "valid_epoch", "valid_iter"]
    y_label = "MSE loss"
    fig_1 = plt.figure(1, figsize=(13, 4))

    ax = plt.subplot(1, 4, 1)
    ax.set_title("train_epoch_RMSE_Etot_loss")
    # ax.set_yscale('log')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(y_label)
    ax.plot(train_epoch_loss["RMSE_Etot"].values, label="RMSE_Etot")

    ax = plt.subplot(1, 4, 2)
    ax.set_title("train_epoch_RMSE_F_loss")
    # ax.set_yscale('log')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(y_label)
    ax.plot(train_epoch_loss["RMSE_F"].values, label="RMSE_F")

    ax = plt.subplot(1, 4, 1)
    ax.set_title("train_iter_RMSE_Etot_loss")
    # ax.set_yscale('log')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(y_label)
    ax.plot(train_iter_loss["RMSE_Etot"].values, label="RMSE_Etot")

    ax = plt.subplot(1, 4, 2)
    ax.set_title("train_iter_RMSE_F_loss")
    # ax.set_yscale('log')
    ax.set_xlabel(x_labels[0])
    ax.set_ylabel(y_label)
    ax.plot(train_iter_loss["RMSE_F"].values, label="RMSE_F")

    cnt += 1
    ax.grid()
    ax.legend()

    plt.savefig(os.path.join(al_pm.opt_log_dir, "train_epoch_iter_loss.png"))

def draw_multi_comp_loss():
    p_dicts = {}
    data_dir = "./mlff_wu_work_dir/data_dir"
    p_1 = get_etot(data_dir)
    p_dicts["p_1"] = p_1
    p_5 = get_etot(os.path.join(data_dir, 'test_best_lkalman_ekf_dir_p_10_5/act_lea_log_dir'))
    p_dicts['p_L5'] = p_5
    p_5_3 = get_etot(os.path.join(data_dir, 'test_best_lkalman_ekf_dir_p_10_5_10_3/act_lea_log_dir'))
    p_dicts['p_L5_L3'] = p_5_3
    p_5_4 = get_etot(os.path.join(data_dir, 'test_best_lkalman_ekf_dir_p_10_5_10_4/act_lea_log_dir'))
    p_dicts['p_L5_L4'] = p_5_4

    del_file(picture_save_dir)

    draw_epoch_loss_train(p_dicts)
    draw_epoch_loss_valid(p_dicts)
    draw_iter_train_loss(p_dicts)
    draw_iter_valid_loss(p_dicts)

if __name__ == "__main__":
    draw_retrain_epoch_iter_train_valid()

