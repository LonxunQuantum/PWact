#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import subprocess
import yaml
import os,sys
import json
import copy
import glob

from active_learning.train.train_util import split_train_dir


from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam

from utils.format_input_output import make_train_name, get_seed_by_time
from utils.constant import AL_STRUCTURE, TRAIN_INPUT_PARAM, TRAIN_FILE_STRUCTUR, MODEL_CMD

from utils.file_operation import search_mvm_files, save_json_file, write_to_file, mv_dir, del_dir
from utils.slurm_script import CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, CONDA_ENV, get_slurm_job_run_info, set_slurm_comm_basis


from active_learning.workdir import WorkTrainDir
from utils.format_input_output import make_iter_name
from active_learning.kpu_util import get_kpu_lower, select_image
from active_learning.slurm import SlurmJob, JobStatus, Mission
from active_learning.pre_al_data_util import get_image_nums

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
    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.resouce = resource
        self.input_param = input_param

        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.train)

    def generate_feature(self):
        feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        # make gen_feature.json file
        train_dict = self.set_train_input_dict(is_mvm=True)
        train_json_file_path = os.path.join(feature_path, TRAIN_FILE_STRUCTUR.feature_json)
        save_json_file(train_dict, train_json_file_path)        
        #make gen_feature.job file
        tag_path = os.path.join(feature_path, TRAIN_FILE_STRUCTUR.feature_tag)
        slrum_gen_feat_script = self.set_train_script(TRAIN_FILE_STRUCTUR.feature_dir, train_json_file_path, tag_path, MODEL_CMD.gen_feat)
        slurm_job_file_path = os.path.join(feature_path, TRAIN_FILE_STRUCTUR.feature_job)
        write_to_file(slurm_job_file_path, slrum_gen_feat_script, "w")        
    
    def do_gen_feature_work(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.train_dir, \
            job_patten="{}/{}".format(TRAIN_FILE_STRUCTUR.feature_dir, TRAIN_FILE_STRUCTUR.feature_job), \
            tag_patten="{}/{}".format(TRAIN_FILE_STRUCTUR.feature_dir, TRAIN_FILE_STRUCTUR.feature_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done is False:
            if len(slurm_remain) > 0:
                slurm_cmd = "sbatch {} ".format(slurm_remain[0])
                slurm_job = SlurmJob()
                tag_path = os.path.join(os.path.dirname(slurm_remain[0]), TRAIN_FILE_STRUCTUR.feature_tag)
                slurm_job.set_tag(tag_path)
                slurm_job.set_cmd(slurm_cmd)
                mission = Mission()
                mission.add_job(slurm_job)
                mission.commit_jobs()
                mission.check_running_job()
            
            assert(mission.all_job_finished())
    
            self.do_post_process_gen_feature()
    
    def do_post_process_gen_feature(self):
        #1. copy feature from work_dir to features_path
        target_feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
        source_feature_path = os.path.join(target_feature_path, TRAIN_FILE_STRUCTUR.work_dir, TRAIN_FILE_STRUCTUR.feature_dir)
        if os.path.exists(source_feature_path):
            mv_dir(source_feature_path, target_feature_path)
        del_dir(os.path.join(target_feature_path, TRAIN_FILE_STRUCTUR.work_dir))
        
    def make_train_work(self):
        for model_index in range(0, self.input_param.strategy.model_num):
            # make model_i work dir
            model_i = make_train_name(model_index)
            model_i_dir = os.path.join(self.train_dir, model_i)
            if not os.path.exists(model_i_dir):
                os.makedirs(model_i_dir)
            
            # make train.json file
            train_dict = self.set_train_input_dict(is_mvm=False)
            train_json_file_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_json)
            save_json_file(train_dict, train_json_file_path)

            # make train slurm script
            tag_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_tag)
            slrum_train_script = self.set_train_script(model_i, train_json_file_path, tag_path, MODEL_CMD.train)
            slurm_job_file_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_job)
            write_to_file(slurm_job_file_path, slrum_train_script, "w")
    
    def get_train_sub_dir(self):
        train_sub_dir = []
        for model_index in self.input_param.strategy.model_num:
            model_i = make_train_name(model_index)
            model_i_dir = os.path.join(self.train_dir, model_i)
            train_sub_dir.append(model_i_dir)
        return train_sub_dir

    '''
    description: 
        If the user provides train.json, use it directly; \
            otherwise, use the user's input settings if available, otherwise use the default values
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def set_train_input_dict(self, is_mvm:True):
        mvm_files = []
        train_feature_path = []
        if is_mvm:
            mvm_files = search_mvm_files(self.input_param.root_dir, self.itername)
            for init_mvm in self.input_param.train.init_mvm_files:
                if os.path.exists(init_mvm):
                    mvm_files.append(init_mvm)
        else:
            train_feature_path = [os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)]
            
        if self.input_param.train.train_input_file is not None and \
            os.path.exists(self.input_param.train.train_input_file):
            # copy user input train.json file
            train_json = json.load(open(self.input_param.train.train_input_file))
            # set train data
        else:
            train_json = self.input_param.train.base_input_param
        
        if is_mvm > 0:
            train_json[TRAIN_INPUT_PARAM.train_mvm_files] = mvm_files
            train_json[TRAIN_INPUT_PARAM.train_feature_path] = []
        else:
            train_json[TRAIN_INPUT_PARAM.train_mvm_files] = []
            train_json[TRAIN_INPUT_PARAM.train_feature_path] = train_feature_path
        if TRAIN_INPUT_PARAM.reserve_feature not in train_json.keys():
            train_json[TRAIN_INPUT_PARAM.reserve_feature] = False
        if TRAIN_INPUT_PARAM.reserve_work_dir not in train_json.keys():
            train_json[TRAIN_INPUT_PARAM.reserve_work_dir] = False
        if TRAIN_INPUT_PARAM.seed not in train_json.keys():
            train_json[TRAIN_INPUT_PARAM.seed] = get_seed_by_time()
        return train_json

    '''
    description: 
    param {*} self
    param {str} job_name
    param {str} train_json
    param {str} tag
    param {str} work_type
        'train' for training, 'gen_feat' for feature generation
    return {*}
    author: wuxingxing
    '''
    def set_train_script(self, job_name:str, train_json:str, tag:str, work_type:str="train"):
        # set head
        job_name = job_name.replace(".","")
        script = ""
        if self.resouce.train_resource.gpu_per_node is None:
            script += CPU_SCRIPT_HEAD.format(job_name, 1, 1, self.resouce.train_resource.queue_name)
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, 1, 1, 1, 1, self.resouce.train_resource.queue_name)

        script += set_slurm_comm_basis(self.resouce.train_resource.custom_flags, \
            self.resouce.train_resource.source_list, \
                self.resouce.train_resource.module_list)
        # set conda env
        script += "\n"
        script += CONDA_ENV

        script += "\n\n"
        script += "PWMLFF {} {}\n\n".format(work_type, train_json)
        script += "test $? -ne 0 && exit 1\n\n"
        script += "echo 0 > {}\n\n".format(tag)
        script += "\n"
        return script

    def do_train_job(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.train_dir, \
            job_patten="*/{}".format(TRAIN_FILE_STRUCTUR.train_job), \
            tag_patten="*/{}".format(TRAIN_FILE_STRUCTUR.train_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done == False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("recover these train Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_cmd = "sbatch {}".format(script_path)
                    slurm_job = SlurmJob()
                    tag = os.path.join(os.path.dirname(script_path),TRAIN_FILE_STRUCTUR.train_tag)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd)
                    mission.add_job(slurm_job)
            else:
                # generate new slurm jobs
                train_sub_dir = self.get_train_sub_dir()
                for train_sub in enumerate(train_sub_dir):
                    success_tag = os.path.join(train_sub, TRAIN_FILE_STRUCTUR.train_tag)
                    script_path = os.path.join(train_sub,TRAIN_FILE_STRUCTUR.train_job)
                    slurm_cmd = "sbatch {}".format(script_path)
                    slurm_job = SlurmJob()
                    slurm_job.set_tag(success_tag)
                    slurm_job.set_cmd(slurm_cmd)
                    mission.add_job(slurm_job)
            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                assert(mission.all_job_finished())
            
            self.post_process_train()
        
    def post_process_train(self):
        if self.input_param.reserve_feature is False:
            feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
            del_dir(feature_path)


    #####################################"""KPU code"""
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