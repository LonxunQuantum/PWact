#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import subprocess
import yaml
import os,sys
import json
import copy
import glob

from active_learning.workdir import WorkTrainDir
from active_learning.util import make_iter_name
from active_learning.kpu_util import get_kpu_lower, select_image
from active_learning.slurm import SlurmJob, JobStatus, Mission
from active_learning.pre_al_data_util import get_image_nums
from active_learning.train_util import split_train_dir, get_kpu_slurm_scripts

'''
description: model training method:
1. go to train data path
2. set training command
3. training
4. post process
param {*} itername
param {*} work_type
return {*}
'''

class ModelTrian(object):
    def __init__(self, itername):
        self.itername = itername
        self.system_info = json.load(open(sys.argv[1]))
        self.train_config = self.system_info["train_config"]
        self.work_root_dir = self.system_info["work_root_path"]
        self.work_dir = WorkTrainDir("{}/{}/training".format(self.work_root_dir, itername))
        
        self.iter_result = json.load(open(os.path.join(self.work_root_dir, "iter_result.json"))) \
            if os.path.exists(os.path.join(self.work_root_dir, "iter_result.json")) is True \
                else {}
        
        self.kpu_res_json = json.load(open(os.path.join(self.work_root_dir, "kpu_result.json"))) \
            if os.path.exists(os.path.join(self.work_root_dir, "kpu_result.json")) is True\
                else {}

    '''
    Description: 
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def make_train(self):
        #1. get train data paths
        data_paths, retain_sign = self.get_feature_data_path()
        if retain_sign is False:
            self.copy_model_from_pre_iter()
            return
        #2. read train_config.yaml file
        with open(self.train_config["config_yaml"], "r") as yamlfile:
            config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
        #3. write data paths to train_config.yaml
        config_yaml["data_paths"] = data_paths

        #4. save to train work dir
        with open(self.work_dir.config_yaml_path, "w") as f:
            yaml.dump(config_yaml, f)
        
        #5. make script and run bash
        if os.path.exists(self.work_dir.train_success_tag) is False:
            if "slurm" in self.system_info.keys():
                script_save_path = self.make_train_cmd_script_slurm()
                slurm_cmd = "sbatch {} ".format(script_save_path)
                slurm_job = SlurmJob()
                slurm_job.set_tag(self.work_dir.train_success_tag)
                slurm_job.set_cmd(slurm_cmd)
                mission = Mission()
                mission.add_job(slurm_job)
                mission.commit_jobs()
                mission.check_running_job()

        #6. calculate kpu of training data
        data_index = "0"
        train_kpu_success_tag = os.path.join(self.work_dir.model_dir, "train_kpu_success_{}.tag".format(data_index))
        if os.path.exists(train_kpu_success_tag) is True:
            return
        if "slurm" in self.system_info.keys():
            #3. generate kpu calculated script file and run
            script_save_path = os.path.join(self.work_dir.work_dir, "train_kpu_slurm_{}.job".format(data_index))
            frq = self.system_info["train_config"]["f"] if "f" in self.system_info["train_config"].keys() else 10
            script_save_path, tag = self.make_kpu_cmd_script_slurm(self.work_dir.config_yaml_path, script_save_path, train_kpu_success_tag, \
                                                      "train", data_index, frq)
            slurm_cmd = "sbatch {}".format(script_save_path)
            slurm_job = SlurmJob()
            slurm_job.set_tag(tag)
            slurm_job.set_cmd(slurm_cmd)
            mission = Mission()
            mission.add_job(slurm_job)
            mission.commit_jobs()
            mission.check_running_job()

    def make_kpu(self):
        mission = Mission()
        #1. get data path
        # get slurm script if exsit.
        slurm_jobs, res_tags, res_done = get_kpu_slurm_scripts(self.work_dir.work_dir)
        kpu_done = True if len(slurm_jobs) == 0 and len(res_done) > 0 else False
        if kpu_done == False:
            #recover slurm jobs
            if len(slurm_jobs) > 0:
                print("recover these KPU Jobs:\n")
                print(slurm_jobs)
                for i, script_save_path in enumerate(slurm_jobs):
                    slurm_cmd = "sbatch {}".format(script_save_path)
                    slurm_job = SlurmJob()
                    slurm_job.set_tag(res_tags[i])
                    slurm_job.set_cmd(slurm_cmd)
                    mission.add_job(slurm_job)
            # generate new slurm jobs
            else:
                data_paths = "{}/{}/exploring/{}".format(self.work_root_dir, self.itername, "md_dpkf_dir")
                data_res = split_train_dir(data_paths)
                
                for v in data_res:
                    data_index = v.split('/')[-1].split('_')[-1]
                    #2. write data paths to train_config.yaml
                    with open(self.work_dir.config_yaml_path, "r") as yamlfile:
                        config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
                    config_yaml["data_paths"] = [v]
                    md_kpu_config_yaml_path = os.path.join(self.work_dir.work_dir, "md_kpu_{}.yaml".format(data_index))
                    with open(md_kpu_config_yaml_path, "w") as f:
                        yaml.dump(config_yaml, f)

                    #3. generate kpu calculated script file and run
                    md_kpu_success_tag = os.path.join(self.work_dir.model_dir, "md_kpu_success_{}.tag".format(data_index))
                    script_save_path = os.path.join(self.work_dir.work_dir, "md_kpu_slurm_{}.job".format(data_index))

                    if os.path.exists(md_kpu_success_tag) is False:
                        if "slurm" in self.system_info.keys():
                            script_save_path, tag = self.make_kpu_cmd_script_slurm(md_kpu_config_yaml_path, script_save_path, md_kpu_success_tag, \
                                                                                            "md", data_index)
                            slurm_cmd = "sbatch {}".format(script_save_path)
                            slurm_job = SlurmJob()
                            slurm_job.set_tag(md_kpu_success_tag)
                            slurm_job.set_cmd(slurm_cmd)
                            mission.add_job(slurm_job)
                        
            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                assert(mission.all_job_finished())

        #5. select images
        select_image(self.system_info, self.work_dir, self.itername)
        
    def make_train_cmd_script_slurm(self):
        with open(os.path.join("./template_script_head", "train.job"), 'r') as rf:
            script_head = rf.readlines()
        for i in script_head:
            if "gres=gpu" in i:
                gpu_nums = int(i.split(":")[-1])
        cmd = ""
        for i in script_head:
            cmd += i
        # gpu_list = self.train_config["gpu_list"] # not used, only used when the machine is single server.
        optimizer = self.train_config["opt"]
        batch_size = self.train_config["b"]
        train_epoch = self.train_config["epochs"]
        train_script_path =self.train_config["train_script_path"]
        # gpus = ""
        # for i in gpu_list:
        #     gpus += "{},".format(i)
        # gpus = gpus[:-1] #gpus: '0,1,2,3,'
        tag = self.work_dir.train_success_tag

        cmd += "\n"
        # cmd += "export CUDA_VISIBLE_DEVICES={}\n".format(gpus)
        cmd_tmp = "python {} --opt {} --epochs {} -r -c {}"
        cmd += cmd_tmp.format(train_script_path, optimizer, train_epoch, self.work_dir.config_yaml_path)
        cmd += " -s {} ".format(self.work_dir.model_dir)
        cmd += " -b {} ".format(batch_size)
        cmd += "\n"
        cmd += "test $? -ne 0 && exit 1\n\n"
        cmd += "echo 0 > {}\n\n".format(tag)
        cmd += "\n"

        with open(self.work_dir.train_script, "w") as wf:
            wf.write(cmd)
        return self.work_dir.train_script

    '''
    Description: 
        make run script for model training
    Author: WU Xingxing
    '''
    def make_train_cmd_script(self):
        gpu_list = self.train_config["gpu_list"]
        # gpus = len(gpu_list)
        optimizer = self.train_config["opt"]
        batch_size = self.train_config["b"]
        train_epoch = self.train_config["epochs"]
        kpu_script_path =self.train_config["kpu_script_path"]
        f = self.train_config["f"] if "f" in self.train_config.keys() else 1
        
        cmd_tmp = "python {} --opt {} --epochs {} -r -c {}"
        cmd = cmd_tmp.format(kpu_script_path, optimizer, train_epoch, self.work_dir.config_yaml_path)
        cmd += " -s {} ".format(self.work_dir.model_dir)
        cmd += " -b {} ".format(batch_size)
        gpus = ""
        for i in gpu_list:
            gpus += "{},".format(i)
        gpus = gpus[:-1] #gpus: '0,1,2,3,'
        tag = self.work_dir.train_success_tag
        res = "\n"
        res += "#!/bin/bash -l\n"
        res += "conda activate mlff_al\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "export CUDA_VISIBLE_DEVICES={}\n".format(gpus)
        
        res += cmd
        
        res += "\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "echo 0 > {}\n\n".format(tag)
        res += "\n"

        with open(self.work_dir.train_script, "w") as wf:
            wf.write(res)

    '''
    Description: 
    param {*} self
    param {*} tag
    param {*} script_save_path
    param {*} config_yaml_path
    param {*} frq: kpu gap
    param {*} index
    Returns: 
    Author: WU Xingxing
    '''
    def make_kpu_cmd_script_slurm(self, config_yaml_path, script_save_path, tag, kpu_type, index="", frq = 1):
        with open(os.path.join("./template_script_head", "kpu.job"), 'r') as rf:
            script_head = rf.readlines()

        # gpu_list = self.train_config["gpu_list"]
        optimizer = self.train_config["opt"]
        train_epoch = self.train_config["epochs"]
        kpu_script_path =self.train_config["kpu_script_path"]
        cmd = ""
        for i in script_head:
            if "--job-name" in i:
                #SBATCH --job-name=kpu_wxx
                cmd += "#SBATCH --job-name=kpu_{}\n".format(index)
            else:
                cmd += i

        cmd += "\n"
        cmd += "test $? -ne 0 && exit 1\n\n"
        # cmd += "export CUDA_VISIBLE_DEVICES={}\n".format(gpu_list[0])

        cmd_tmp = "python {} --opt {} --epochs {} -r -c {}"
        cmd += cmd_tmp.format(kpu_script_path, optimizer, train_epoch, config_yaml_path)
        cmd += " -s {} ".format(self.work_dir.model_dir)
        cmd += " -b 1 "
        cmd += " -k "
        cmd += " -f {} ".format(frq)
        kpu_type = "{}_{}_kpu_dir".format(kpu_type, index)
        cmd += " --kpu_dir {}".format(kpu_type)
        
        cmd += "\n"
        cmd += "test $? -ne 0 && exit 1\n\n"
        cmd += "echo 0 > {}\n\n".format(tag)
        cmd += "\n"

        with open(script_save_path, "w") as wf:
            wf.write(cmd)
        return script_save_path, tag
    
    '''
    Description: 
    param {*} self
    param {*} kpu_type is "md":for mlff md image kpu, is "train": for training data kpu.
    Returns: 
    Author: WU Xingxing
    '''
    def make_kpu_cmd_script(self, kpu_type="md"):
        if kpu_type == "md":
            tag = self.work_dir.md_kpu_success_tag
            script_save_path = self.work_dir.md_kpu_script
            config_yaml_path = self.work_dir.md_kpu_config_yaml_path
            f = 1
        else:
            tag = self.work_dir.train_kpu_success_tag
            script_save_path = self.work_dir.train_kpu_script
            config_yaml_path = self.work_dir.config_yaml_path
            f = self.train_config["f"] if "f" in self.train_config.keys() else 1

        gpu_list = self.train_config["gpu_list"]
        optimizer = self.train_config["opt"]
        train_epoch = self.train_config["epochs"]
        train_sript_path =self.train_config["train_sript_path"]
        
        cmd_tmp = "python {} --opt {} --epochs {} -r -c {}"
        cmd = cmd_tmp.format(train_sript_path, optimizer, train_epoch, config_yaml_path)
        cmd += " -s {} ".format(self.work_dir.model_dir)
        cmd += " -b 1 "
        cmd += " -k "
        cmd += " -f {} ".format(f)
        kpu_type = "{}_kpu_dir".format(kpu_type)
        cmd += " --kpu_dir {}".format(kpu_type)
        
        res = "\n"
        res += "#!/bin/bash -l\n"
        res += "conda activate mlff_al\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "export CUDA_VISIBLE_DEVICES={}\n".format(gpu_list[0])
        
        res += cmd
        
        res += "\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "echo 0 > {}\n\n".format(tag)
        res += "\n"

        with open(script_save_path, "w") as wf:
            wf.write(res)

    """
    @Description :
    get training data path
    if first training: return init_data
    else: return feature_path of pre-iter labeling result
    @Returns     :
    @Author       :wuxingxing
    """
    def get_feature_data_path(self): #no use
        retain_sign = False
        iter_index = int(self.itername[5:])
        if iter_index == 0:
            return self.system_info["init_data_path"], True

        pre_iter_name = make_iter_name(iter_index-1)
        if self.iter_result[pre_iter_name]["retrain"] is False:
            return [], False
        
        res = []
        keys = list(self.iter_result.keys()) #反向
        keys = sorted(keys, key=lambda x:int(x[5:]))
        tmp = []
        for i in keys:
            if self.iter_result[i]["retrain"] is True:
                tmp.append(self.iter_result[i]["feature_path"])
                retain_sign = True
        if len(tmp) <= 6:
            res.extend(self.system_info["init_data_path"])
        res.extend(tmp)
        return res, retain_sign

    '''
    Description: 
    if not enough data to train, link the training dir of pre-iter as current inter model
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def copy_model_from_pre_iter(self):
        older_iter_name = make_iter_name(int(self.itername[5:]) - 1)
        pre_work_dir = WorkTrainDir("{}/{}/training".format(self.work_root_dir, older_iter_name))
        #link model
        if os.path.exists(self.work_dir.model_save_path) is False:
            os.symlink(pre_work_dir.model_save_path, self.work_dir.model_save_path)
        
        #link train kpu
        if os.path.exists(self.work_dir.train_kpu_dir) is False:
            os.symlink(pre_work_dir.train_kpu_dir, self.work_dir.train_kpu_dir)
        
        #copy train.yaml
        if os.path.exists(self.work_dir.config_yaml_path) is False:
            shutil.copy(pre_work_dir.config_yaml_path, self.work_dir.config_yaml_path)
            
        #copy tag train_kpu_success.tag  train_success.tag
        if os.path.exists(self.work_dir.train_success_tag) is False:
            shutil.copy(pre_work_dir.train_success_tag, self.work_dir.train_success_tag)
        source_train_kpu_success_tag = glob.glob(os.path.join(pre_work_dir.model_dir, "train_kpu_success*.tag"))
        target_train_kpu_success_tag = os.path.join(self.work_dir.model_dir, os.path.basename(source_train_kpu_success_tag[0]))
        if os.path.exists(self.work_dir.train_kpu_success_tag) is False:
            shutil.copy(source_train_kpu_success_tag[0], self.work_dir.train_kpu_success_tag)

        #set kpu limits according to result of pre-iteration
        self.kpu_res_json[self.itername] = {}
        self.kpu_res_json[self.itername]["force_kpu_lower"] = self.kpu_res_json[older_iter_name]["force_kpu_lower"]
        self.kpu_res_json[self.itername]["force_kpu_upper"] = self.kpu_res_json[older_iter_name]["force_kpu_upper"]
        # self.kpu_res_json[self.itername]["etot_kpu_lower"] = self.kpu_res_json[older_iter_name]["etot_kpu_lower"]
        # self.kpu_res_json[self.itername]["etot_kpu_upper"] = self.kpu_res_json[older_iter_name]["etot_kpu_upper"]

        json.dump(self.kpu_res_json, open(os.path.join(self.system_info["work_root_path"], "kpu_result.json"), "w"), indent=4)
        print("==========model training {} have been done!=======".format(self.work_dir.work_dir))
        return