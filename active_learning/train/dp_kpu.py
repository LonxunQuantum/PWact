"""
    dir of explore:
    iter.0000/explore/md:
    --------------------/*-md.job file
    --------------------/*-md.tag.success file
    --------------------/md.000.sys.000/ dir
    -----------------------------------/md.000.sys.000.t.000.p.000 or md.000.sys.000.t.000 dir
    -------------------------------------------------------------/md files: lmp.config, in.lammps, forcefile files, model_devi file
    -------------------------------------------------------------/trajs
    iter.0000/explore/select: this is kpu work dir
    -------------------/md.000.sys.000/
    ----------------------------------/md.000.sys.000.t.000(p.000)
    -------------------------------------------------------------/kpu.job kpu.json dp_model.ckpt
    -------------------------------------------------------------kpu dir
    --------------------------------------------------------------------------/kpu_info.csv
    after select
    -------------------/summary.txt
    -------------------/accurate.txt candidate.txt failed.txt candidate_del.txt
    
    the content of candidate.txt is:
"""
import shutil
import subprocess
import yaml
import os,sys
import json
import copy
import glob
import numpy as np
import pandas as pd
import random

from active_learning.slurm import SlurmJob, JobStatus, Mission, get_slurm_sbatch_cmd
from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam

from utils.format_input_output import make_train_name, get_seed_by_time, get_iter_from_iter_name, get_sub_md_sys_template_name
from utils.constant import AL_STRUCTURE, TRAIN_INPUT_PARAM, TRAIN_FILE_STRUCTUR, MODEL_CMD, FORCEFILED, UNCERTAINTY, \
    EXPLORE_FILE_STRUCTURE, LABEL_FILE_STRUCTURE

from utils.file_operation import save_json_file, write_to_file, mv_dir, del_dir, link_file, search_files
from utils.slurm_script import CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, CONDA_ENV, get_slurm_job_run_info, set_slurm_comm_basis, split_job_for_group
from utils.app_lib.pwmat import convert_config_to_mvm

from utils.format_input_output import make_iter_name

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

class ModelKPU(object):
    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.resouce = resource
        self.input_param = input_param

        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.train)

        # md work dir
        self.iter = get_iter_from_iter_name(self.itername)
        self.sys_paths = self.input_param.explore.sys_configs
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)

    def make_kpu_work(self):
        model_i = make_train_name(0) # train/train.000
        model_i_dir = os.path.join(self.train_dir, model_i)
        model_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)
        # get mvm files
        kpu_work_list = []
        mvm_md_list = self.get_mvm_in_md()
        # for each mvm file, make a kpu calculate dir
        for mvm_index, mvm in enumerate(mvm_md_list):
            #1. make kpu work dir
            # md/md.*.sys.*/md.*.sys.*.t.*.p* or md.*.sys.*.t.*
            md_sys_t_p = os.path.dirname(mvm)
            # md/md.*.sys.*
            md_sys = os.path.dirname(md_sys_t_p)
            #select/md.*.sys.*/md.*.sys.*.t.*.p* or md.*.sys.*.t.*
            kpu_work_dir = os.path.join(self.select_dir, os.path.basename(md_sys), os.path.basename(md_sys_t_p))
            if not os.path.exists(kpu_work_dir):
                os.makedirs(kpu_work_dir)
            #2. make kpu.json file
            kpu_dict = self.set_kpu_input_dict([mvm], model_path)
            kpu_json_file_path = os.path.join(kpu_work_dir, TRAIN_FILE_STRUCTUR.kpu_json)
            save_json_file(kpu_dict, kpu_json_file_path)
            #3. link model
            # target_model_path = os.path.join(kpu_work_dir, TRAIN_FILE_STRUCTUR.dp_model_name)
            # link_file(model_path, target_model_path)
            
            kpu_work_list.append(kpu_work_dir)
            
        # make base kpu dir, randomly select 20% mvms from active learning iter.*/label/result
        mvm_files = search_files(self.input_param.root_dir, "iter.*/{}/{}/mvm-*-".format(AL_STRUCTURE.labeling, LABEL_FILE_STRUCTURE.result))
        mvm_files.extend(self.input_param.train.init_mvm_files)
        if len(mvm_files) > 10:
            percent = self.input_param.strategy.base_kpu_mvm_ratio
            mvm_files = random.sample(mvm_files, int(len(mvm_files) * percent))
        kpu_dict = self.set_kpu_input_dict(mvm_files, model_path, sample_nums=self.input_param.strategy.base_kpu_max_images)
        std_kpu_dir = os.path.join(self.select_dir, base_kpu)
        if not os.path.exists(std_kpu_dir):
            os.makedirs(std_kpu_dir)
        kpu_json_file_path = os.path.join(std_kpu_dir, TRAIN_FILE_STRUCTUR.kpu_json)
        save_json_file(kpu_dict, kpu_json_file_path)
        # target_model_path = os.path.join(std_kpu_dir, TRAIN_FILE_STRUCTUR.dp_model_name)
        # link_file(model_path, target_model_path)
        kpu_work_list.append(kpu_work_dir)
        # make train slurm script
        self.make_kpu_slurm_jobs(kpu_work_list)
    
    def make_kpu_slurm_jobs(self, kpu_work_list:list[str]):
        #4. set slurm job file
        group_list = split_job_for_group(self.resouce.explore_resource.group_size, kpu_work_list)
        for g_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "kpu{}".format(g_index)
            tag_name = "{}-{}".format(g_index, TRAIN_FILE_STRUCTUR.kpu_tag)
            tag = os.path.join(self.select_dir, tag_name)
            slurm_job_script = self.set_kpu_slurm_job_script(group, jobname, tag, TRAIN_FILE_STRUCTUR.kpu_json)
            slurm_script_name = "{}-{}".format(g_index, EXPLORE_FILE_STRUCTURE.md_job)
            slurm_job_file = os.path.join(self.md_dir, slurm_script_name)
            write_to_file(slurm_job_file, slurm_job_script, "w")

    def get_mvm_in_md(self):
        #1. get md.*.sys.*/md.*.sys.*. dirs
        md_name_tamp = get_sub_md_sys_template_name()
        md_sys_list = glob.glob(os.path.join(self.md_dir, md_name_tamp))
        
        #2. for each dir, convert the *.config to mvm.md.*.sys.*.abs
        mvm_list = []
        for md_sys in md_sys_list:
            config_temp = "{}/{}".format("config", "*.config")
            config_list = glob.glob(os.path.join(md_sys, config_temp))
            config_list = sorted(config_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
            # convert configs in md.*.sys.*/md.*.sys.*.t.* to mvm-md.*.sys.*.t.*
            mvm_save_file = "{}/mvm-{}".format(md_sys, os.path.basename(md_sys))
            convert_config_to_mvm(config_list, mvm_save_file)
            mvm_list.append(mvm_save_file)
        return mvm_list
        
    '''
    description: 
        If the user provides train.json, use it directly; \
            otherwise, use the user's input settings if available, otherwise use the default values
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def set_kpu_input_dict(self, mvms:list[str], model_path:str, sample_nums:int=None):
        train_json = self.input_param.train.get_train_input_dict()
        train_json[TRAIN_INPUT_PARAM.test_mvm_files] = mvms
        train_json[TRAIN_INPUT_PARAM.train_feature_path] = []       
        train_json[TRAIN_INPUT_PARAM.model_load_file] = model_path
        if sample_nums is not None:
            train_json[TRAIN_INPUT_PARAM.sample_nums] = sample_nums
        train_json[TRAIN_INPUT_PARAM.test_dir_name] = TRAIN_FILE_STRUCTUR.kpu
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
    def set_kpu_slurm_job_script(self, group:list[str], job_name:str, tag:str, train_json:str):
        # set head
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
        script += "\n"
        script += "start=$(date +%s)\n"
        
        kpu_cmd = ""
        for kpu_dir in group:
            if kpu_dir == "NONE":
                continue
            kpu_cmd += "cd {}\n".format(kpu_dir)
            kpu_cmd += "if [ ! -f {} ] ; then\n".format(TRAIN_FILE_STRUCTUR.kpu_tag)
            kpu_cmd += "    PWMLFF {} {}\n\n".format(MODEL_CMD.kpu, train_json)
            kpu_cmd += "    if test $? -eq 0; then touch {}; else echo 1 >> {}; fi\n".format(TRAIN_FILE_STRUCTUR.kpu_tag, TRAIN_FILE_STRUCTUR.kpu_tag_failed)
            kpu_cmd += "fi &\n"
            kpu_cmd += "wait\n\n"
            
        script += kpu_cmd
        script += "end=$(date +%s)\n"
        script += "take=$(( end - start ))\n"
        script += "echo Time taken to execute commands is ${{take}} seconds > {}\n".format(tag)
        script += "\n"
        
        return script

        '''
    description: 
        waiting: if need set group size, make new script: work1 wait; work2 wait; ...
    param {*} self
    param {list} md_work_list
    return {*}
    author: wuxingxing
    '''    
    def do_kpu_jobs(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.select_dir, \
            job_patten="*-{}".format(TRAIN_FILE_STRUCTUR.kpu_job), \
            tag_patten="*-{}".format(TRAIN_FILE_STRUCTUR.kpu_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done == False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Doing these KPU Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_cmd = get_slurm_sbatch_cmd(os.path.dirname(script_path), os.path.basename(script_path))
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), TRAIN_FILE_STRUCTUR.kpu_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd, os.path.dirname(script_path))
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
                mission.move_slurm_log_to_slurm_work_dir()
                    
    '''
    description: 
        iter.0000/explore/select: this is kpu work dir
        -------------------/md.000.sys.000/
        ----------------------------------/md.000.sys.000.t.000(p.000)
        -------------------------------------------------------------/kpu.job kpu.json dp_model.ckpt
        -------------------------------------------------------------kpu dir
        --------------------------------------------------------------------------/kpu_info.csv
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def post_process_kpu(self):
        # 1. find kpu_info.csv files
        devi_pd = pd.DataFrame(columns=["devi_force", "file_path", "config_index"])
        kpu_info_patten =  "{}/{}/{}".format(get_sub_md_sys_template_name(), TRAIN_FILE_STRUCTUR.kpu, TRAIN_FILE_STRUCTUR.kpu_file)
        kpu_info_files = glob.glob(os.path.join(self.md_dir, kpu_info_patten))
        # 2. load datas
        for kpu_file in kpu_info_files:
            devi_force = pd.read_csv(kpu_file)
            tmp_pd = pd.DataFrame()
            tmp_pd["devi_force"] = devi_force["force_kpu"]
            tmp_pd["config_index"] = devi_force["step"]
            tmp_pd["file_path"] = os.path.dirname(kpu_file)
            devi_pd = pd.concat([devi_pd, tmp_pd])

        # 3. select images with lower and upper limitation
        
        lower = self.get_lower_from_base_kpu()*self.input_param.strategy.kpu_lower
        higer = lower * self.input_param.strategy.kpu_upper
        max_select = self.input_param.strategy.max_select
        accurate_pd  = devi_pd[devi_pd['devi_force'] < lower]
        candidate_pd = devi_pd[devi_pd['devi_force'] >= lower and devi_pd['devi_force'] < higer]
        error_pd     = devi_pd[devi_pd['devi_force'] > higer]
        #4. if selected images more than number limitaions, randomly select
        remove_candi = None
        rand_candi = None
        if candidate_pd.shape[0] > max_select:
            rand_candi = candidate_pd.sample(max_select)
            remove_candi = candidate_pd.drop(rand_candi.index)
        
        #5. save select info
        accurate_pd.to_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.accurate))
        candi_info = ""
        if rand_candi is not None:
            rand_candi.to_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
            remove_candi.to_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate_delete))
            candi_info += "candidate configurations: {}, randomly select {}, delete {}\n\
                \t select details in file {}\n\t delete details in file {}.".format(
                    candidate_pd.shape[0], rand_candi.shape[0], remove_candi.shape[0],\
                    os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate),\
                    os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate_delete)  
                )
        else:
            candidate_pd.to_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
            candi_info += "candidate configurations: {}\n\t select details in file {}\n".format(
                    candidate_pd.shape[0],
                    os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
                
        error_pd.to_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.failed))
        
        summary_info = ""
        summary_info += "total configurations: {}\n".format(devi_pd.shape[0])
        summary_info += "select by model deviation force:\n"
        summary_info += "accurate configurations: {}, details in file {}\n".\
            foramt(accurate_pd.shape[0], os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.accurate))
            
        summary_info += candi_info
            
        summary_info += "error configurations: {}, details in file {}\n".\
            foramt(error_pd.shape[0], os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.failed))
        
        write_to_file(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.select_summary), summary_info)
        print("committee method result:\n {}".format(summary_info))
        
    def get_lower_from_base_kpu(self):
        base_kpu = os.path.join(self.select_dir, TRAIN_FILE_STRUCTUR.base_kpu, TRAIN_FILE_STRUCTUR.kpu_file)
        base = pd.read_csv(base_kpu)
        return base["force_kpu"].mean()
