#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from active_learning.slurm import SlurmJob, Mission
from utils.slurm_script import CHECK_TYPE, \
    get_slurm_job_run_info, set_slurm_script_content
from active_learning.user_input.resource import Resource
from active_learning.user_input.iter_input import InputParam

from utils.format_input_output import make_train_name, get_seed_by_time, get_iter_from_iter_name
from utils.constant import AL_STRUCTURE, TEMP_STRUCTURE, TRAIN_INPUT_PARAM, TRAIN_FILE_STRUCTUR, MODEL_CMD, FORCEFILED, LABEL_FILE_STRUCTURE

from utils.file_operation import save_json_file, write_to_file, del_dir, search_files, file_shell_op, link_file

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
        self.train_dir = os.path.join(self.input_param.root_dir, self.itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.train)
        self.real_train_dir = os.path.join(self.input_param.root_dir, self.itername, AL_STRUCTURE.train)
    
    def make_train_work(self):
        for model_index in range(0, self.input_param.strategy.model_num):
            # make model_i work dir
            model_i = make_train_name(model_index)
            model_i_dir = os.path.join(self.train_dir, model_i)
            if not os.path.exists(model_i_dir):
                os.makedirs(model_i_dir)
            
            # make train.json file
            train_dict = self.set_train_input_dict(is_mvm=True, work_dir=model_i_dir)
            train_json_file_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_json)
            save_json_file(train_dict, train_json_file_path)

            # make train slurm script
            jobname = "train{}".format(model_index)
            tag_name = TRAIN_FILE_STRUCTUR.train_tag
            tag = os.path.join(model_i_dir, tag_name)
            run_cmd = self.set_train_cmd(train_json_file_path)
            
            train_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.train_resource.gpu_per_node, 
                number_node = self.resource.train_resource.number_node, 
                cpu_per_node = self.resource.train_resource.cpu_per_node,
                queue_name = self.resource.train_resource.queue_name,
                custom_flags = self.resource.train_resource.custom_flags,
                source_list = self.resource.train_resource.source_list,
                module_list = self.resource.train_resource.module_list,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = [model_i_dir],
                job_tag = tag,
                task_tag = TRAIN_FILE_STRUCTUR.train_tag, 
                task_tag_faild = TRAIN_FILE_STRUCTUR.train_tag_failed,
                parallel_num=1,
                check_type=CHECK_TYPE.train
                )
            slurm_job_file_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_job)
            write_to_file(slurm_job_file_path, train_slurm_script, "w")

    def set_train_cmd(self, train_json:str):
        model_path = "./{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)
        cmp_model_path = None
        script = ""
        script += "PWMLFF {} {}\n\n".format(MODEL_CMD.train, os.path.basename(train_json))
        if self.input_param.strategy.compress:
            script += "    PWMLFF {} {} -d {} -o {} -s {}\n\n".format(MODEL_CMD.compress, model_path, \
                self.input_param.strategy.compress_dx, self.input_param.strategy.compress_order, TRAIN_FILE_STRUCTUR.compree_dp_name)
            cmp_model_path = "{}".format(TRAIN_FILE_STRUCTUR.compree_dp_name)
            
        if self.input_param.strategy.md_type == FORCEFILED.libtorch_lmps:
            if cmp_model_path is None:
                script += "    PWMLFF {} {}\n\n".format(MODEL_CMD.script, model_path)
            else:
                script += "    PWMLFF {} {}\n\n".format(MODEL_CMD.script, cmp_model_path)
        return script

    def get_train_sub_dir(self):
        train_sub_dir = []
        for model_index in self.input_param.strategy.model_num:
            model_i = make_train_name(model_index)
            model_i_dir = os.path.join(self.train_dir, model_i)
            train_sub_dir.append(model_i_dir)
        return train_sub_dir

    '''
    description: 
        make train json dict content
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def set_train_input_dict(self, is_mvm:bool=True, work_dir:str=None):
        mvm_files = []
        train_feature_path = []
        if is_mvm:
            mvm_files = search_files(self.input_param.root_dir, "iter.*/{}/mvm-*-".format(LABEL_FILE_STRUCTURE.result))
            for init_mvm in self.input_param.init_mvm_files:
                if os.path.exists(init_mvm):
                    mvm_files.append(init_mvm)
        else:
            train_feature_path = [os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)]
        
        train_json = self.input_param.train.to_dict()
        # reset seed and work_dir
        train_json[TRAIN_INPUT_PARAM.work_dir] = os.path.join(work_dir, "work_dir")
        train_json[TRAIN_INPUT_PARAM.seed] = get_seed_by_time()
        if is_mvm > 0:
            train_json[TRAIN_INPUT_PARAM.train_mvm_files] = mvm_files
            train_json[TRAIN_INPUT_PARAM.train_feature_path] = []
        else:
            train_json[TRAIN_INPUT_PARAM.train_mvm_files] = []
            train_json[TRAIN_INPUT_PARAM.train_feature_path] = train_feature_path
        # if TRAIN_INPUT_PARAM.reserve_feature not in train_json.keys():
        train_json[TRAIN_INPUT_PARAM.reserve_feature] = True
        # if TRAIN_INPUT_PARAM.reserve_work_dir not in train_json.keys():
        train_json[TRAIN_INPUT_PARAM.reserve_work_dir] = True
        return train_json

    def do_train_job(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.train_dir, \
            job_patten="*/{}".format(TRAIN_FILE_STRUCTUR.train_job), \
            tag_patten="*/{}".format(TRAIN_FILE_STRUCTUR.train_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("recover these train Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag = os.path.join(os.path.dirname(script_path),TRAIN_FILE_STRUCTUR.train_tag)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()

    def post_process_train(self):
        temp_work_dirs = search_files(self.train_dir, "*/{}".format("work_dir"))
        if self.input_param.reserve_work is True:
            pass
        else:
            for temp in temp_work_dirs:
                del_dir(temp)
            # if self.input_param.reserve_feature is False:
            #     feature_path = os.path.join(self.train_dir, TRAIN_FILE_STRUCTUR.feature_dir)
            #     del_dir(feature_path)
        link_file(self.train_dir, self.real_train_dir)
