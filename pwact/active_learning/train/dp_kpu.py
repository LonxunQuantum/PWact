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
import os
import glob
import numpy as np
import pandas as pd

from pwact.active_learning.slurm.slurm import SlurmJob, Mission
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.iter_input import InputParam
from pwact.active_learning.explore.select_image import select_image
from pwact.utils.format_input_output import make_train_name, get_iter_from_iter_name, get_sub_md_sys_template_name,\
    get_md_sys_template_name
from pwact.utils.constant import AL_STRUCTURE, TRAIN_FILE_STRUCTUR, MODEL_CMD, \
    EXPLORE_FILE_STRUCTURE, LAMMPS, SLURM_OUT, TEMP_STRUCTURE, UNCERTAINTY

from pwact.utils.file_operation import write_to_file, link_file, search_files, del_file_list_by_patten
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
from pwact.active_learning.explore.select_image import select_image
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
        self.resource = resource
        self.input_param = input_param
        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.train)

        # md work dir
        self.iter = get_iter_from_iter_name(self.itername)
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        # md work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.kpu_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.kpu) # for kpu calculate
      
        self.real_explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.real_md_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.real_select_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.real_kpu_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.kpu) # for kpu calculate
 
        
    def make_kpu_work(self):
        kpu_list = []
        
        model_i = make_train_name(0) # train/train.000
        model_i_dir = os.path.join(self.train_dir, model_i)
        source_model_path = os.path.join(model_i_dir, TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.dp_model_name)

        md_sys_dir_list = search_files(self.md_dir, get_md_sys_template_name())
        for md_sys_dir in md_sys_dir_list:
            sub_md_sys_dir_list =search_files(md_sys_dir, get_md_sys_template_name())
            for sub_md_sys in sub_md_sys_dir_list: #kpu_work_dir
                # set kpu work
                # link kpu calculate model
                kpu_dir = os.path.join(self.kpu_dir, os.path.basename(md_sys_dir), os.path.basename(sub_md_sys))
                if not os.path.exists(kpu_dir):
                    os.makedirs(kpu_dir)
                
                link_file(source_model_path, os.path.join(kpu_dir, TRAIN_FILE_STRUCTUR.dp_model_name))
                link_file(os.path.join(sub_md_sys, LAMMPS.atom_type_file), os.path.join(kpu_dir, LAMMPS.atom_type_file))
                link_file(os.path.join(sub_md_sys, EXPLORE_FILE_STRUCTURE.traj), os.path.join(kpu_dir, EXPLORE_FILE_STRUCTURE.traj))
                kpu_list.append(kpu_dir)
        
        self.make_kpu_slurm_job_files(kpu_list)

    def make_kpu_slurm_job_files(self, kpu_list:list[str]):
        del_file_list_by_patten(self.kpu_dir, "*{}".format(TRAIN_FILE_STRUCTUR.kpu_job))
        group_list = split_job_for_group(self.resource.train_resource.group_size, kpu_list, 1)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "kpu{}".format(group_index)
            tag_name = "{}-{}".format(group_index, TRAIN_FILE_STRUCTUR.kpu_tag)
            tag = os.path.join(self.kpu_dir, tag_name)
            run_cmd = self.set_kpu_cmd()
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
                task_tag = TRAIN_FILE_STRUCTUR.kpu_tag, 
                task_tag_faild = TRAIN_FILE_STRUCTUR.kpu_tag_failed,
                parallel_num=1,
                check_type=None
                )
            slurm_script_name = "{}-{}".format(group_index, TRAIN_FILE_STRUCTUR.kpu_job)
            slurm_job_file = os.path.join(self.kpu_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def set_kpu_cmd(self):
        script = ""
        pwmlff = self.resource.train_resource.command
        script += "{} {} -m {} -c {} -f {} -a {} -s {} >> {}\n\n".format(\
            pwmlff, MODEL_CMD.kpu, \
            TRAIN_FILE_STRUCTUR.dp_model_name, EXPLORE_FILE_STRUCTURE.traj, \
            LAMMPS.traj_format, LAMMPS.atom_type_file, EXPLORE_FILE_STRUCTURE.kpu_model_devi, SLURM_OUT.kpu_out)
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
        slurm_remain, slurm_success = get_slurm_job_run_info(self.kpu_dir, \
            job_patten="*-{}".format(TRAIN_FILE_STRUCTUR.kpu_job), \
            tag_patten="*-{}".format(TRAIN_FILE_STRUCTUR.kpu_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these KPU Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), TRAIN_FILE_STRUCTUR.kpu_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.kpu_out)
                # mission.move_slurm_log_to_slurm_work_dir()
                    
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
        summary = select_image(
                md_dir=self.kpu_dir, 
                save_dir=self.select_dir,
                md_job=self.md_job,
                devi_name=EXPLORE_FILE_STRUCTURE.get_devi_name(UNCERTAINTY.kpu),
                lower=self.input_param.strategy.kpu_lower,  
                higer=self.input_param.strategy.kpu_upper
        )
        return summary

