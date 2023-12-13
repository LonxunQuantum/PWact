from email import header
from genericpath import isdir
import os
import json
import numpy as np
import pandas as pd
import re
import logging
import random
"""
@Description :
#由权重、协方差分布重新生成权重 目前未使用
@Returns     :
@Author       :wuxingxing
"""
def gen_multivariate_normal(mean_weight, cov_P):
    # w0 = weights[0].cpu().detach().numpy()
    # wP = self.P[i].cpu().detach().numpy()
    return np.random.multivariate_normal(mean=mean_weight,cov=cov_P)

'''
@File         :util.py
@Description  :封装logging为单例对象
@Time         :2022/06/29 14:39:09
@Author       :wuxingxing
'''
class LogFrame(object):
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)#LogFrame, cls
        return cls.__instance

    def __init__(self, logging_level_DUMP, logging_level_SUMMARY, opt_log_level, opt_logging_file, opt_file_log_level, logger_path) -> None:
        self.logging_level_DUMP = logging_level_DUMP
        self.logging_level_SUMMARY = logging_level_SUMMARY
        
        logging.addLevelName(self.logging_level_DUMP, 'DUMP')
        logging.addLevelName(self.logging_level_SUMMARY, 'SUMMARY')

        self.logger = logging.getLogger(logger_path)
        self.logger.setLevel(self.logging_level_DUMP)

        formatter = logging.Formatter("\33[0m\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
        handler1 = logging.StreamHandler()
        handler1.setLevel(opt_log_level)
        handler1.setFormatter(formatter)
        self.logger.addHandler(handler1)

        if (opt_logging_file != ''):
            formatter = logging.Formatter("\33[0m\33[32;49m[%(asctime)s]\33[0m.\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
            handler2 = logging.FileHandler(filename = opt_logging_file)
            handler2.setLevel(opt_file_log_level)
            handler2.setFormatter(formatter)
            self.logger.addHandler(handler2)

    def dump(self, msg, *args, **kwargs):
        self.logger.log(self.logging_level_DUMP, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def summary(self, msg, *args, **kwargs):
        self.logger.log(self.logging_level_SUMMARY, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs, exc_info=True)   

    """
    @Description :
    打印程序运行初试界面
    @Returns     :
    @Author       :wuxingxing
    """
    def show_start_logging_banner_to_logging_file(self, info):
        # show start logging banner to logging file
        self.summary("")
        self.summary("#########################################################################################")
        self.summary("#            ___          __                         __      __  ___       __  ___      #")
        self.summary("#      |\ | |__  |  |    |__) |  | |\ | |\ | | |\ | / _`    /__`  |   /\  |__)  |       #")
        self.summary("#      | \| |___ |/\|    |  \ \__/ | \| | \| | | \| \__>    .__/  |  /~~\ |  \  |       #")
        self.summary("#                                                                                       #")
        self.summary("#########################################################################################")
        self.summary("")
        self.summary(' '.join(info))
        self.summary("")

    """
    @Description :
    打印初始参数集合    
    @Returns     :
    @Author       :wuxingxing
    """
    def print_args_used_in_ekf(self, opts, momentum, REGULAR_wd, n_epoch, LR_base, LR_gamma, LR_step, batch_size):
        self.info("Training: session = %s" %opts.opt_session_name)
        self.info("Training: run_id = %s" %opts.opt_run_id)
        self.info("Training: journal_cycle = %d" %opts.opt_journal_cycle)
        self.info("Training: follow_mode = %s" %opts.opt_follow_mode)
        self.info("Training: recover_mode = %s" %opts.opt_recover_mode)
        self.info("Training: network = %s" %opts.opt_net_cfg)
        self.info("Training: model_dir = %s" %opts.opt_model_dir)
        self.info("Training: model_file = %s" %opts.opt_model_file)
        self.info("Training: activation = %s" %opts.opt_act)
        self.info("Training: optimizer = %s" %opts.opt_optimizer)
        self.info("Training: momentum = %.16f" %momentum)
        self.info("Training: REGULAR_wd = %.16f" %REGULAR_wd)
        self.info("Training: scheduler = %s" %opts.opt_scheduler)
        self.info("Training: n_epoch = %d" %n_epoch)
        self.info("Training: LR_base = %.16f" %LR_base)
        self.info("Training: LR_gamma = %.16f" %LR_gamma)
        self.info("Training: LR_step = %d" %LR_step)
        self.info("Training: batch_size = %d" %batch_size)

        # scheduler specific options
        self.info("Scheduler: opt_LR_milestones = %s" %opts.opt_LR_milestones)
        self.info("Scheduler: opt_LR_patience = %s" %opts.opt_LR_patience)
        self.info("Scheduler: opt_LR_cooldown = %s" %opts.opt_LR_cooldown)
        self.info("Scheduler: opt_LR_total_steps = %s" %opts.opt_LR_total_steps)
        self.info("Scheduler: opt_LR_max_lr = %s" %opts.opt_LR_max_lr)
        self.info("Scheduler: opt_LR_min_lr = %s" %opts.opt_LR_min_lr)
        self.info("Scheduler: opt_LR_T_max = %s" %opts.opt_LR_T_max)
        self.info("scheduler: opt_autograd = %s" %opts.opt_autograd)

"""
@Description :
读取最后一行文件
@Returns     :
@Author       :wuxingxing
"""
def file_read_last_line(file_path, type_name="int"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            last_line = rf.readlines()[-1]  #the last line
            if '[]' in last_line:
                return []
            last_line = last_line.replace(" ","")[1:-2].split(',')
    if len(last_line) > 0 and type_name == "int":
        last_line = [int(i) for i in last_line]
    if len(last_line) > 0 and type_name == "float":
        last_line = [float(i) for i in last_line]
    return last_line

def file_read_lines(file_path, type_name="float"):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            lines = rf.readlines()  #the last line
            for line in lines:
                line = re.sub('[\[\]\\n]','',line)
                if len(line) > 0 and type_name == "int":
                    line = [int(i) for i in line.split(',')]
                if len(line) > 0 and type_name == "float":
                    line = [float(i) for i in line.split(',')]
                data.append(line)
    return data

"""
@Description :
存储行 array 数据到txt最后一行
@Returns     :
@Author       :wuxingxing
"""
def write_to_file(file_path, line):
    with open(file_path, 'a') as wf:
        wf.write(line)


"""
@Description :
删除指定目录下所有文件(该目录不删除) / 或者删除指定文件名文件
@Returns     :
@Author       :wuxingxing
"""

def del_file(path_dir):
    if os.path.exists(path_dir) == False:
        return

    if os.path.isfile(path_dir):
        os.remove(path_dir)
        return
        
    for i in os.listdir(path_dir) :
        file_path = os.path.join(path_dir, i)
        if os.path.isfile(file_path) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_path)
        else:
            del_file(file_path)
    
def cal_index(localtion, train_dfeat):
    config = []
    config_dict = {}
    with open(localtion, "r") as rf:
        lines = rf.readlines()
        for line in lines:
            if "PWdata/" in line:
                config.append(line.split("/")[-2]+'/'+line.split("/")[-1][:-1])
    for config in config:
        image_index = 0
        start = None
        end = 0
        for image in train_dfeat:
            image_sys = image[0].split("/")[-3] + "/" + image[0].split("/")[-2]
            image_num = image[1]
            if image_sys == config:
                start = image_index if start is None else start
                end = image_index
            image_index += 1
        config_dict[config] = [start, end]
    return config_dict

"""
@Description :
获取体系中训练/测试数据 image与movement文件对应关系
@Returns     :
@Author       :wuxingxing
"""

def get_image_index(system):
    train_path = "/data/data/wuxingxing/datas/{}/fread_dfeat/NN_output/dfeatname_train.csv1".format(system)
    valid_path = "/data/data/wuxingxing/datas/{}/fread_dfeat/NN_output/dfeatname_test.csv1".format(system)
    train_dfeat = pd.read_csv(train_path, header=None, encoding= 'unicode_escape').values
    valid_dfeat = pd.read_csv(valid_path, header=None, encoding= 'unicode_escape').values
    localtion = "/data/data/wuxingxing/datas/{}/PWdata/location".format(system)
    train_image_index = cal_index(localtion, train_dfeat)
    valid_image_index = cal_index(localtion, valid_dfeat)
    print(train_image_index)
    print(valid_image_index)
    return train_image_index, valid_image_index

"""
@Description :
查找所有的dpkfmovement目录路径
@Returns     :/data/data/wuxingxing/codespace/dpgen_v2/gen_work/de770385839a54fe40725abfccfcfbacb468c098/data.init/CH4.POSCAR.01x01x01/02.md/sys-0004-0001
@Author       :wuxingxing
"""

def search_dpkfmovement_dirs(path_name, search_path, data_paths):
    dirs = os.listdir(search_path)
    for dir in dirs:
        if os.path.isdir(os.path.join(search_path, dir)):
            if dir == path_name:
                data_paths.append(os.path.join(search_path, dir))
            else:
                data_paths = search_dpkfmovement_dirs(path_name, os.path.join(search_path, dir), data_paths) 
    return data_paths

def get_recent_model(model_dir):
    if os.path.exists(model_dir) is False:
        return False, None, None
    epoch = 0
    reload = False
    model_path = None
    p_path = None
    file_list = os.listdir(model_dir)
    model_file_list = []
    for i in file_list:
        if "checkpoint" in i:
            model_file_list.append(i)
    if len(model_file_list) > 0:
        model_file_list = sorted(model_file_list, key=lambda file: int(file.split('_')[2]), reverse=True)
        model_path = os.path.join(model_dir, model_file_list[0])
        p_path = os.path.join(model_dir, "epoch_{}P.pt".format(model_file_list[0].split('_')[2]))
        reload = True
    return reload, model_path, p_path

def test_search():
    search_path = "/data/data/wuxingxing/codespace/dpgen_v2/gen_work/de770385839a54fe40725abfccfcfbacb468c098"
    data_path = search_dpkfmovement_dirs("dpkfmovement", search_path, [])
    print(data_path)

def make_iter_name (iter_index) :
    iter_format = "%04d"
    return "iter." + (iter_format % iter_index)

def combine_files(source_dir=None, source_files=None, target_file=None):
    with open(target_file, 'w') as outfile:
        for file in source_files:
            infile_path = os.path.join(source_dir, file) if source_dir is not None else file
            with open(infile_path, 'r') as infile:     
                outfile.write(infile.read()) 
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

"""
@Description :
    write all movement of iters to one movement file 
@Returns     :
@Author       :wuxingxing
"""

def write_iter_result_to_movements(iter_path, target_file):
    iters = json.load(open(iter_path))
    iterlist = list(iters.keys())
    movement_list = set()
    for key in iterlist:
        movement_dir = iters[key]["movement_dir"]
        movement_file_list = iters[key]["movement_file"]
        if len(movement_file_list) > 0:
            for file in movement_file_list:
                movement_list.add(os.path.join(movement_dir, file))
    combine_files(source_files = list(movement_list), target_file = target_file)

def read_iter_result(iter_path):
    iters = json.load(open(iter_path))
    iterlist = list(iters.keys())
    
    for key in iterlist:
        res = []
        if "movement_file" in iters[key].keys():
            res.extend([int(_.split("-")[1]) for _ in iters[key]["movement_file"]])
        res = sorted(res)
        print("{}:{}".format(key, res))

"""
@Description :
    randomly generate n different nums of int type in the range of [start, end)
@Returns     :
@Author       :wuxingxing
"""

def get_random_nums(start, end, n):
    random.seed(2022)
    numsArray = set()
    while len(numsArray) < n:
        numsArray.add(random.randint(start, end-1))
    return list(numsArray)

if __name__ == "__main__":
    # train_image_index, valid_image_index = get_image_index("li")
    # test_search()

    # logging_level_DUMP = 5 
    # logging_level_SUMMARY = 15
    # opt_log_level =logging.INFO
    # opt_logging_file = '/home/wuxingxing/codespace/MLFF_wu_dev/init_model_ekf_dir/refactoring_log'
    # opt_file_log_level = logging.DEBUG
    # logs = LogFrame(logging_level_DUMP, logging_level_SUMMARY, opt_log_level, opt_logging_file, opt_file_log_level)

    # logs.info(" a refactor test ")
    # logs.debug("debug info ")
    # import sys
    # logs.show_start_logging_banner_to_logging_file((sys.argv))
    # print()
    # save_as_line(al_pm.batch_cadidate_path, [1,2, 4, 5])
    # ary = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117
    # , 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135
    # , 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

    # # print(str(ary))
    # # save_as_line(al_pm.batch_cadidate_path, str(ary))
    # # save_as_line(al_pm.batch_cadidate_path, str(ary))
    # print(file_read_last_line(al_pm.batch_accurate_path, "int"))
    # file_read_last_line(al_pm.batch_accurate_path, "int")
    # res = get_random_nums(0, 60, 30)
    # print(res)

    # avg_images_kpu, all_images_kpu = read_kpu_data("/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/slab_1300K_all/log_dir/30_v2_epoch_kpu_train", "30_v2_epoch_kpu_train_kpu_force.csv")
    # kpu_mean, kpu_min, kpu_max, kpu_med = avg_images_kpu[["kpu", "kpu_min", "kpu_max", "kpu_med"]].mean()
    # print(avg_images_kpu[["kpu", "kpu_min", "kpu_max", "kpu_med"]][:450].mean())
    
    # avg_images_kpu1, all_images_kpu1 = read_kpu_data("/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/slab_1300K_all/log_dir/30_v3_all_epoch_kpu_train", "30_v3_all_epoch_kpu_train_kpu_force.csv")
    # print(avg_images_kpu1[["kpu", "kpu_min", "kpu_max", "kpu_med"]][:750].mean())

    # from draw_pictures.draw_util import draw_distribution
    # save_path = "/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/slab_1300K_all/log_dir/30_v2_epoch_kpu_train_kpu_force_distribution.png"
    # draw_distribution(avg_images_kpu["kpu"][:450], save_path, title="KPU_f distribution of training set")

    # save_path = "/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/slab_1300K_all/log_dir/30_v3_all_epoch_kpu_train_kpu_force_distribution.png"
    # draw_distribution(avg_images_kpu1["kpu"][:750], save_path, title="KPU_f distribution of valid set")

    target_file = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/iter.0003/test/final_MOVEMENT"
    write_iter_result_to_movements("/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/iter_result.json", target_file)

    # read_iter_result("/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cu_slab_1500k_system/iter_result.json")