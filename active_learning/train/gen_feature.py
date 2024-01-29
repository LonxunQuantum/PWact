#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from active_learning.slurm import SlurmJob, Mission
from utils.slurm_script import get_slurm_job_run_info, set_slurm_script_content
from active_learning.user_input.resource import Resource
from active_learning.user_input.iter_input import InputParam
from utils.format_input_output import get_iter_from_iter_name, get_md_sys_template_name
from utils.constant import AL_STRUCTURE, TEMP_STRUCTURE, TRAIN_FILE_STRUCTUR, LABEL_FILE_STRUCTURE, \
    TRAIN_INPUT_PARAM, MODEL_CMD, PWMAT

from utils.file_operation import save_json_file, write_to_file, search_files

'''
description: 
    gen feature: the movement from this iteration after scf work
param {*} itername
param {*} work_type
return {*}
'''

class GenFormatNpy(object):
    def __init__(self, work_dir:str, 
                        mvm_list:list[str], 
                        train_valid_ratio:float=0.8, 
                        data_shuffle:bool=True,
                        queue_name=None
                        ):
        self.work_dir = work_dir
        self.mvm_list = mvm_list
        self.train_valid_ratio = train_valid_ratio
        self.valid_shuffle = data_shuffle

    def check_state(self):
        slurm_remain, slurm_done = get_slurm_job_run_info(self.work_dir, \
            job_patten=TRAIN_FILE_STRUCTUR.feature_job, \
            tag_patten=TRAIN_FILE_STRUCTUR.feature_tag)
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        return slurm_done

    def make_gen_work(self):
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        #1. make gen_feature.json file
        train_dict = self.set_train_input_dict()
        train_json_file_path = os.path.join(self.work_dir, TRAIN_FILE_STRUCTUR.feature_json)
        save_json_file(train_dict, train_json_file_path)

        #2. make train slurm script
        jobname = "genfeat"
        tag_name = TRAIN_FILE_STRUCTUR.feature_tag
        tag = os.path.join(self.work_dir, tag_name)
        run_cmd = self.set_gen_feat_cmd(train_json_file_path)
        train_slurm_script = set_slurm_script_content(gpu_per_node=None, 
            number_node = 1, 
            cpu_per_node = 1,
            queue_name = self.resource.train_resource.queue_name,
            custom_flags = self.resource.train_resource.custom_flags,
            source_list = self.resource.train_resource.source_list,
            module_list = self.resource.train_resource.module_list,
            job_name = jobname,
            run_cmd_template = run_cmd,
            group = [self.work_dir],
            job_tag = tag,
            task_tag = TRAIN_FILE_STRUCTUR.feature_tag, 
            task_tag_faild = TRAIN_FILE_STRUCTUR.feature_tag_failed,
            parallel_num=1,
            check_type=None
            )
        slurm_job_file_path = os.path.join(self.work_dir, TRAIN_FILE_STRUCTUR.feature_job)
        write_to_file(slurm_job_file_path, train_slurm_script, "w")
        
    def set_gen_feat_cmd(self, train_json:str):
        script = ""
        script += "{} {}\n\n".format(MODEL_CMD.pwdata, os.path.basename(train_json))
        return script

    def do_gen_work(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.work_dir, \
            job_patten=TRAIN_FILE_STRUCTUR.feature_job, \
            tag_patten=TRAIN_FILE_STRUCTUR.feature_tag)
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done is False:
            if len(slurm_remain) > 0:
                print("recover this data format change Job:\n")
                print(slurm_remain)
                slurm_job = SlurmJob()
                tag_path = os.path.join(os.path.dirname(slurm_remain[0]), TRAIN_FILE_STRUCTUR.feature_tag)
                slurm_job.set_tag(tag_path)
                slurm_job.set_cmd(slurm_remain[0])
                mission = Mission()
                mission.add_job(slurm_job)
            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
    
    '''
    description: 
        If the user provides train.json, use it directly; \
            otherwise, use the user's input settings if available, otherwise use the default values
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def set_train_input_dict(self):
        train_json = {}
        train_json[TRAIN_INPUT_PARAM.train_valid_ratio] = self.train_valid_ratio
        train_json[TRAIN_INPUT_PARAM.valid_shuffle] = self.valid_shuffle
        train_json[TRAIN_INPUT_PARAM.raw_files] = self.mvm_list
        train_json[TRAIN_INPUT_PARAM.datasets_path] = []
        return train_json