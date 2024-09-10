#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
from pwact.active_learning.slurm.slurm import SlurmJob, Mission, scancle_job
from pwact.utils.slurm_script import get_slurm_job_run_info, set_slurm_script_content, split_job_for_group
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.iter_input import InputParam

from pwact.utils.format_input_output import make_train_name, get_seed_by_time, get_iter_from_iter_name, make_iter_name
from pwact.utils.constant import AL_STRUCTURE, UNCERTAINTY, TEMP_STRUCTURE, MODEL_CMD, \
    TRAIN_INPUT_PARAM, TRAIN_FILE_STRUCTUR, FORCEFILED, LABEL_FILE_STRUCTURE, SLURM_OUT, MODEL_TYPE

from pwact.utils.file_operation import save_json_file, write_to_file, del_dir, search_files, add_postfix_dir, mv_file, copy_dir, del_file_list, del_file_list_by_patten
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
    @staticmethod
    def kill_job(root_dir:str, itername:str):
        train_dir =  os.path.join(root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.train)
        scancle_job(train_dir)

    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.resource = resource
        self.input_param = input_param
        self.iter = get_iter_from_iter_name(self.itername)
        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, self.itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.train)
        self.real_train_dir = os.path.join(self.input_param.root_dir, self.itername, AL_STRUCTURE.train)
    
    '''
    description: 
        if the slurm jobs done
        it means this iter done before, need back up train files in iter*/train directory 
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def back_train(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.real_train_dir, \
            job_patten="*-{}".format(TRAIN_FILE_STRUCTUR.train_job), \
            tag_patten="*-{}".format(TRAIN_FILE_STRUCTUR.train_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        if slurm_done:
            # bk and do new job
            target_bk_file = add_postfix_dir(self.real_train_dir, postfix_str="bk")
            mv_file(self.real_train_dir, target_bk_file)
            # if the temp_work_dir/train exists, delete the train dir
            if os.path.exists(self.train_dir):
                del_dir(self.train_dir)
                
    def make_train_work(self):
        train_list = []
        for model_index in range(0, self.input_param.strategy.model_num):
            # make model_i work dir
            model_i = make_train_name(model_index)
            model_i_dir = os.path.join(self.train_dir, model_i)
            if not os.path.exists(model_i_dir):
                os.makedirs(model_i_dir)
            # make train.json file
            train_dict = self.set_train_input_dict(work_dir=model_i_dir)
            train_json_file_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.train_json)
            save_json_file(train_dict, train_json_file_path)
            train_list.append(model_i_dir)
        self.make_train_slurm_job_files(train_list)
    
    def make_train_slurm_job_files(self, train_list:list[str]):
        # make train slurm script
        del_file_list_by_patten(self.train_dir, "*{}".format(TRAIN_FILE_STRUCTUR.train_job))
        group_list = split_job_for_group(self.resource.train_resource.group_size, train_list, 1)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue

            jobname = "train{}".format(group_index)
            tag_name = "{}-{}".format(group_index, TRAIN_FILE_STRUCTUR.train_tag)
            tag = os.path.join(self.train_dir, tag_name)
            run_cmd = self.set_train_cmd()

            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.train_resource.gpu_per_node, 
                number_node = self.resource.train_resource.number_node, 
                cpu_per_node = self.resource.train_resource.cpu_per_node,
                queue_name = self.resource.train_resource.queue_name,
                custom_flags = self.resource.train_resource.custom_flags,
                env_script = self.resource.train_resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = TRAIN_FILE_STRUCTUR.train_tag, 
                task_tag_faild = TRAIN_FILE_STRUCTUR.train_tag_failed,
                parallel_num=1,
                check_type=None
                )
            slurm_script_name = "{}-{}".format(group_index, TRAIN_FILE_STRUCTUR.train_job)
            slurm_job_file = os.path.join(self.train_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def set_train_cmd(self):
        model_path = "./{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)
        cmp_model_path = None
        script = ""
        pwmlff = self.resource.train_resource.command
        script += "{} {} {} >> {}\n\n".format(pwmlff, MODEL_CMD.train, TRAIN_FILE_STRUCTUR.train_json, SLURM_OUT.train_out)

        # do nothing for nep model
        if self.input_param.train.model_type == MODEL_TYPE.dp:
            if self.input_param.strategy.compress:
                script += "    {} {} {} -d {} -o {} -s {}/{} >> {}\n\n".format(pwmlff, MODEL_CMD.compress, model_path, \
                    self.input_param.strategy.compress_dx, self.input_param.strategy.compress_order, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.compree_dp_name, SLURM_OUT.train_out)
                cmp_model_path = "{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.compree_dp_name)
            
            if self.input_param.strategy.md_type == FORCEFILED.libtorch_lmps:
                if self.resource.explore_resource.gpu_per_node is None or self.resource.explore_resource.gpu_per_node == 0:
                    script += "    export CUDA_VISIBLE_DEVICES=''\n"
                if cmp_model_path is None:
                    # script model_record/dp_model.ckpt the torch_script_module.pt will in model_record dir
                    script += "    {} {} {} {}/{} >> {}\n".format(pwmlff, MODEL_CMD.script, model_path, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.script_dp_name, SLURM_OUT.train_out)
                else:
                    script += "    {} {} {} {}/{} >> {}\n\n".format(pwmlff, MODEL_CMD.script, cmp_model_path, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.script_dp_name, SLURM_OUT.train_out)
        return script

    '''
    description: 
        if the init format exists movement files, convert them to npy format and save to root_dir/init_data dir
        then in each iter, the init_data is from the root_dir
        make train json dict content
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def set_train_input_dict(self, work_dir:str=None):
        train_json = self.input_param.train.to_dict()
        train_feature_path = []
        if self.input_param.init_data_only_pretrain and self.iter > 0:
            # use old model param iter.*/train/train.000/model_record/dp_model.ckpt
            pre_model = os.path.join(self.input_param.root_dir, make_iter_name(self.iter-1), \
                AL_STRUCTURE.train, make_train_name(0), TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)
            train_json[TRAIN_INPUT_PARAM.recover_train] = True
            train_json[TRAIN_INPUT_PARAM.model_load_file] = pre_model
            train_json[TRAIN_INPUT_PARAM.optimizer][TRAIN_INPUT_PARAM.reset_epoch] = True
        else:
            for _data in self.input_param.init_data:
                train_feature_path.append(_data)
        # search train_feature_path in iter*/label/result/*/PWdata/*
        iter_index = get_iter_from_iter_name(self.itername)
        start_iter = 0
        while start_iter < iter_index:
            iter_pwdata = search_files(self.input_param.root_dir, 
                                    "{}/{}/{}/*".format(make_iter_name(start_iter), AL_STRUCTURE.labeling, LABEL_FILE_STRUCTURE.result))
            if len(iter_pwdata) > 0:
                train_feature_path.extend(iter_pwdata)
            start_iter += 1
        
        # reset seed
        train_json[TRAIN_INPUT_PARAM.seed] = get_seed_by_time()
        train_json[TRAIN_INPUT_PARAM.raw_files] = []
        train_json[TRAIN_INPUT_PARAM.datasets_path] = train_feature_path
        if self.input_param.strategy.uncertainty == UNCERTAINTY.kpu:
            train_json[TRAIN_INPUT_PARAM.save_p_matrix] = True
        return train_json

    def do_train_job(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.train_dir, \
            job_patten="*-{}".format(TRAIN_FILE_STRUCTUR.train_job), \
            tag_patten="*-{}".format(TRAIN_FILE_STRUCTUR.train_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these train Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), TRAIN_FILE_STRUCTUR.train_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.train_out)

    def post_process_train(self):
        copy_dir(self.train_dir, self.real_train_dir)
        del_file_list(search_files(self.real_train_dir, "slurm*.out"))
        del_file_list(search_files(self.real_train_dir, "*/{}".format(SLURM_OUT.train_out)))
