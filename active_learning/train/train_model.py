#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,sys
import json
import copy

from active_learning.slurm import SlurmJob, Mission, get_slurm_sbatch_cmd
from utils.slurm_script import CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, CONDA_ENV, CHECK_TYPE, \
    get_slurm_job_run_info, set_slurm_comm_basis, set_slurm_script_content
from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam

from utils.format_input_output import make_train_name, get_seed_by_time
from utils.constant import AL_STRUCTURE, TRAIN_INPUT_PARAM, TRAIN_FILE_STRUCTUR, MODEL_CMD, FORCEFILED, LABEL_FILE_STRUCTURE

from utils.file_operation import save_json_file, write_to_file, mv_file, del_dir, search_files

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
        self.resource = resource
        self.input_param = input_param
        self.iter = get_iter_from_iter_name(self.itername)
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
        tag_path = TRAIN_FILE_STRUCTUR.feature_tag
        
        # slrum_gen_feat_script = set_slurm_script_content(gpu_per_node=self.resource.train_resource.gpu_per_node, 
        #                      number_node = self.resource.train_resource.number_node, 
        #                      cpu_per_node = self.resource.train_resource.cpu_per_node,
        #                      queue_name = self.resource.train_resource.queue_name,
        #                      custom_flags = self.resource.train_resource.custom_flags,
        #                      source_list = self.resource.train_resource.source_list,
        #                      module_list = self.resource.train_resource.module_list,
        #                      job_name = TRAIN_FILE_STRUCTUR.feature_dir.replace(".",""),
        #                      run_cmd_template = "PWMLFF {} {}".format(MODEL_CMD.gen_feat, os.path.basename(train_json_file_path)),
        #                      group = [feature_path],
        #                      job_tag = os.path.join(feature_path, tag_path),
        #                      task_tag = tag_path,
        #                      task_tag_faild = feature_tag_failed,
        #                      parallel_num=1
        #                      )
        
        slrum_gen_feat_script = self.set_train_script(TRAIN_FILE_STRUCTUR.feature_dir.replace(".",""), train_json_file_path, tag_path, MODEL_CMD.gen_feat)
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
                slurm_cmd = get_slurm_sbatch_cmd(os.path.dirname(slurm_remain[0]), os.path.basename(slurm_remain[0]))
                slurm_job = SlurmJob()
                tag_path = os.path.join(os.path.dirname(slurm_remain[0]), TRAIN_FILE_STRUCTUR.feature_tag)
                slurm_job.set_tag(tag_path)
                slurm_job.set_cmd(slurm_cmd, os.path.dirname(slurm_remain[0]))
                mission = Mission()
                mission.add_job(slurm_job)
                mission.commit_jobs()
                mission.check_running_job()
            
            mission.all_job_finished()
            # mission.move_slurm_log_to_slurm_work_dir(self.input_param.root_dir)
            self.do_post_process_gen_feature()
    
    def do_post_process_gen_feature(self):
        #1. copy feature from work_dir to features_path
        target_feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
        source_feature_path = os.path.join(target_feature_path, TRAIN_FILE_STRUCTUR.work_dir, TRAIN_FILE_STRUCTUR.feature_dir)
        if os.path.exists(source_feature_path):
            mv_file(source_feature_path, target_feature_path)
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
            tag_path = TRAIN_FILE_STRUCTUR.train_tag
            slrum_train_script = self.set_train_script(model_i.replace(".",""), train_json_file_path, tag_path, MODEL_CMD.train)
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
            mvm_files = search_files(self.input_param.root_dir, "iter.*/{}/mvm-*-".format(LABEL_FILE_STRUCTURE.result))
            for init_mvm in self.input_param.train.init_mvm_files:
                if os.path.exists(init_mvm):
                    mvm_files.append(init_mvm)
        else:
            train_feature_path = [os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)]
        
        train_json = self.input_param.train.get_train_input_dict()
        
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
        script = ""
        if self.resource.train_resource.gpu_per_node is None:
            script += CPU_SCRIPT_HEAD.format(job_name, 1, 1, self.resource.train_resource.queue_name)
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, 1, 1, 1, 1, self.resource.train_resource.queue_name)

        script += set_slurm_comm_basis(self.resource.train_resource.custom_flags, \
            self.resource.train_resource.source_list, \
                self.resource.train_resource.module_list)
        # set conda env
        script += "\n"
        script += CONDA_ENV

        script += "\n\n"
        work_dir = os.path.dirname(train_json)
        script += "cd {}\n".format(work_dir)
        
        script += "start=$(date +%s)\n"
        script += "PWMLFF {} {}\n\n".format(work_type, os.path.basename(train_json))
        script += "test $? -ne 0 && exit 1\n\n"
        
        if work_type == MODEL_CMD.train:
            model_path = "./{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)
            cmp_model_path = None
            if self.input_param.strategy.compress:
                script += "PWMLFF {} {} -d {} -o {} -s {}\n\n".format(MODEL_CMD.compress, model_path, \
                    self.input_param.strategy.compress_dx, self.input_param.strategy.compress_order, TRAIN_FILE_STRUCTUR.compree_dp_name)
                script += "test $? -ne 0 && exit 1\n\n"
                cmp_model_path = "{}".format(TRAIN_FILE_STRUCTUR.compree_dp_name)
                
            if self.input_param.strategy.md_type == FORCEFILED.libtorch_lmps:
                if cmp_model_path is None:
                    script += "PWMLFF {} {}\n\n".format(MODEL_CMD.script, model_path)
                else:
                    script += "PWMLFF {} {}\n\n".format(MODEL_CMD.script, cmp_model_path)
                script += "test $? -ne 0 && exit 1\n\n"
            
        script += "end=$(date +%s)\n"
        script += "take=$(( end - start ))\n"
        script += "echo Time taken to execute commands is ${{take}} seconds > {}\n".format(tag)
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
                    slurm_cmd = get_slurm_sbatch_cmd(os.path.dirname(script_path), os.path.basename(script_path))
                    slurm_job = SlurmJob()
                    tag = os.path.join(os.path.dirname(script_path),TRAIN_FILE_STRUCTUR.train_tag)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd, os.path.dirname(script_path))
                    mission.add_job(slurm_job)
            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
                # mission.move_slurm_log_to_slurm_work_dir(self.input_param.root_dir)
        
    def post_process_train(self):
        if self.input_param.reserve_feature is False:
            feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
            del_dir(feature_path)
