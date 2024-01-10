"""
    dir or label:
    iter.0000/label/scf:
    --------------------/*-scf.job file
    --------------------/*-scf.tag.success file
    --------------------/md.000.sys.000/ dir
    -----------------------------------/md.000.sys.000.t.000.p.000 or md.000.sys.000.t.000 dir
    -------------------------------------------------------------/scf_0/atom.config etot.input pseudo files
    -------------------------------------------------------------/scf_2/atom.config etot.input pseudo files
    -------------------------------------------------------------...
    
    iter.0000/label/result:
    -------------------/summary.txt
    -------------------/md.000.sys.000-mvm, md.000.sys.001-mvm, ...
    
    the content of summary.txt is:
            md.000.sys.000-mvm: image_nums, atom_type
            md.000.sys.001-mvm: image_nums, atom_type
            ...
"""
import os, sys
import numpy as np
import pandas as pd
from math import ceil

from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam, SCFParam
from active_learning.slurm import SlurmJob, Mission, get_slurm_sbatch_cmd

from utils.constant import AL_STRUCTURE, LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, PWMAT
from utils.slurm_script import CONDA_ENV, CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, get_slurm_job_run_info, set_slurm_comm_basis, split_job_for_group
from utils.format_input_output import get_iter_from_iter_name, get_md_sys_template_name
from utils.file_operation import write_to_file, copy_file, link_file, merge_files_to_one
from utils.app_lib.pwmat import make_pwmat_input_dict, set_etot_input_by_file, get_atom_type_from_atom_config

class Labeling(object):
    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.iter = get_iter_from_iter_name(self.itername)
        self.resouce = resource
        self.input_param = input_param
        
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        # md work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)
        # labed work dir
        self.label_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.labeling) 
        self.scf_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.scf)
        self.result_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.result)

    '''
    description: 
    the scf work dir file structure is as follow.
    iter.0000/label/scf:
    --------------------/*-scf.job file
    --------------------/*-scf.tag.success file
    --------------------/md.000.sys.000/ 
    -----------------------------------/md.000.sys.000.t.000.p.000 or md.000.sys.000.t.000 
    -------------------------------------------------------------/0-scf/atom.config etot.input pseudo files
    -------------------------------------------------------------/2-scf/atom.config etot.input pseudo files
    -------------------------------------------------------------...
    param {*} self
    return {*}
    author: wuxingxing
    '''        
    def make_scf_work(self):
        # read select info, and make scf
        # ["devi_force", "file_path", "config_index"]
        candidate = pd.read_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
        # make scf work dir
        scf_dir_list = []
        for index, row in candidate.iterrows():
            file_path       = row["file_path"]
            config_index    = row["config_index"]
            sub_md_sys_path = os.path.dirname(file_path)
            sub_md_sys_name = os.path.basename(sub_md_sys_path)
            md_sys_path     = os.path.dirname(sub_md_sys_path)
            md_sys_name     = os.path.basename(md_sys_path)
            scf_sub_md_sys_path = os.path.join(self.scf_dir, md_sys_name, sub_md_sys_name, "{}-{}".format(config_index, LABEL_FILE_STRUCTURE.scf))
            if not os.path.exists(scf_sub_md_sys_path):
                os.makedirs(scf_sub_md_sys_path)
        
            tarj_dir = os.path.join(temp_dir, EXPLORE_FILE_STRUCTURE.tarj)
            traj_atom_config = os.path.join(sub_md_sys_path, "{}-{}".format(config_index, PWMAT.config_postfix))
            self.make_scf_file(scf_sub_md_sys_path, traj_atom_config)
            scf_dir_list.extend(scf_sub_md_sys_path)
            
        self.make_scf_slurm_job_files(scf_dir_list)

    def do_scf_jobs(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_job), \
            tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done == False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Doing these SCF Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_cmd = get_slurm_sbatch_cmd(os.path.dirname(script_path), os.path.basename(script_path))
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), EXPLORE_FILE_STRUCTURE.md_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd, os.path.dirname(script_path))
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
                # mission.move_slurm_log_to_slurm_work_dir()
                    
    def make_scf_file(self, scf_dir:str, source_config:str):
        #1. link atom.config file
        target_atom_config = os.path.join(scf_dir, PWMAT.atom_config)
        link_file(source_config, target_atom_config)
        #2. copy pseudo potential file
        # from atom.config get atom type
        atom_type_list = get_atom_type_from_atom_config(source_config)
        pseudo_list = []
        for atom in atom_type_list:
            pseudo_atom_path = SCFParam.get_pseudo_by_atom_name(self.input_param.scf.pseudo, atom)
            pseduo_name = os.path.basename(pseudo_atom_path)
            copy_file(pseudo_atom_path, os.path.join(scf_dir, pseduo_name))
            pseudo_list.append(pseduo_name)
        #3. make etot.input file
        etot_script = set_etot_input_by_file(self.input_param.scf.scf_etot_input_file, target_atom_config, [self.resouce.scf_resource.number_node, self.resouce.scf_resource.gpu_per_node])
        
        # if self.input_param.scf.etot_input_file is not None:
        #     etot_script = set_etot_input_by_file(self.input_param.scf.etot_input_file, target_atom_config, [self.resouce.scf_resource.number_node, self.resouce.scf_resource.gpu_per_node])
        # else:
        #     etot_script = make_pwmat_input_dict(
        #     node1 = scfparam.node1,
        #     node2 = scfparam.node2,
        #     job_type = PWMAT.scf,
        #     pseudo_list = pseudo_list,
        #     atom_config = target_atom_config,
        #     ecut = scfparam.ecut,
        #     ecut2 = scfparam.ecut2,
        #     e_error = scfparam.e_error,
        #     rho_error = scfparam.rho_error,
        #     out_force = scfparam.out_force,
        #     energy_decomp = scfparam.energy_decomp,
        #     out_stress = scfparam.out_stress,
        #     icmix = scfparam.icmix,
        #     smearing = scfparam.smearing,
        #     sigma = scfparam.sigma,
        #     kspacing = scfparam.kspacing,
        #     flag_symm = scfparam.flag_symm,
        #     out_wg = scfparam.out_wg,
        #     out_rho = scfparam.out_rho,
        #     out_mlmd = scfparam.out_mlmd,
        #     vdw=scfparam.vdw,
        #     relax_detail=scfparam.relax_detail
        #     )        
        etot_input_file = os.path.join(scf_dir, PWMAT.etot_input)
        write_to_file(etot_input_file, etot_script, "w")

    def make_scf_slurm_job_files(self, scf_sub_list:list[str]):
        pwmat_num = self.resouce.pwmat_run_num
        groupsize = 1 if self.resouce.scf_resource.group_size is None \
            else self.resouce.scf_resource.group_size
        
        if groupsize > 1:
            groupsize_adj = ceil(groupsize/pwmat_num)
            if groupsize_adj*pwmat_num > groupsize:
                groupsize_adj = ceil(groupsize/pwmat_num)*pwmat_num
                print("the groupsize automatically adjusts from {} to {}".format(groupsize, groupsize_adj))
        else:
            groupsize_adj = groupsize
        group_list = split_job_for_group(groupsize_adj, scf_sub_list)
        
        group_script_path = []
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            slurm_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.scf_job)
            tag_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.scf_tag)
            slurm_path = os.path.join(self.scf_dir, slurm_name)
            tag_path = os.path.join(self.scf_dir, tag_name)
            jobname = "scf{}".format(group_index)
            slurm_job_script = self.set_scf_slurm_job_script(group, jobname, tag_path)
            write_to_file(slurm_path, slurm_job_script, "w")
            group_script_path.append(slurm_path)
        return slurm_path
            
    def set_scf_slurm_job_script(self,group:list[str], job_name:str, tag:str):
        # set head
        script = ""
        if self.resouce.scf_resource.gpu_per_node is None:
            script += CPU_SCRIPT_HEAD.format(job_name, \
                self.resouce.scf_resource.number_node,\
                self.resouce.scf_resource.cpu_per_node,\
                    self.resouce.scf_resource.queue_name)
            mpirun_cmd_template = "mpirun -np {} PWmat".format(self.resouce.scf_resource.cpu_per_node)
            
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, \
                self.resouce.scf_resource.number_node,\
                self.resouce.scf_resource.gpu_per_node,\
                    self.resouce.scf_resource.gpu_per_node,\
                    1,\
                    self.resouce.scf_resource.queue_namee)
            mpirun_cmd_template = "mpirun -np {} PWmat".format(self.resouce.scf_resource.gpu_per_node)
                            
        script += set_slurm_comm_basis(self.resouce.scf_resource.custom_flags, \
            self.resouce.scf_resource.source_list, \
                self.resouce.scf_resource.module_list)
        
        # set conda env
        script += "\n"
        script += CONDA_ENV
        
        script += "\n\n"
        
        script += "start=$(date +%s)\n"

        scf_cmd = ""
        job_id = 0
        while job_id < len(group):
            for i in range(self.self.resouce.pwmat_run_num):
                if group[job_id] == "NONE":
                    job_id += 1
                    continue
                scf_cmd += "{\n"
                scf_cmd += "cd {}\n".format(group[job_id])
                scf_cmd += "if [ ! -f {} ] ; then\n".format(LABEL_FILE_STRUCTURE.scf_tag)
                scf_cmd += "    {}\n".format(mpirun_cmd_template)
                scf_cmd += "    if test $? -eq 0; then touch {}; else touch {}; fi\n".format(LABEL_FILE_STRUCTURE.scf_tag, LABEL_FILE_STRUCTURE.scf_tag_failed)
                scf_cmd += "fi\n"
                scf_cmd += "} &\n\n"
                job_id += 1
            scf_cmd += "wait\n\n"
        
        script += scf_cmd
        
        script += "\ntest $? -ne 0 && exit 1\n\n"
        
        script += "end=$(date +%s)\n"
        script += "take=$(( end - start ))\n"
        script += "echo Time taken to execute commands is ${{take}} seconds > {}\n\n".format(tag)
        script += "\n"
        return script

    def post_process_scf(self):
        mvm_list = []
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        md_sys_dir_list = glob.glob(os.path.join(self.scf_dir, get_md_sys_template_name()))
        for md_sys_dir in md_sys_dir_list:
            md_sys_mlmd = []
            sub_md_sys_dir_list = glob.glob(os.path.join(md_sys_dir, get_md_sys_template_name()))
            for sub_md_sys in sub_md_sys_dir_list:
                out_mlmd_list = glob.glob(os.path.join(sub_md_sys, "*-{}/{}".format(LABEL_FILE_STRUCTURE.scf, PWMAT.out_mlmd)))
                # do a sorted?
                md_sys_mlmd.extend(out_mlmd_list)

            mvm_save_file = os.path.join(self.result_dir, "{}-{}-{}".format(PWMAT.mvm, len(md_sys_mlmd), os.path.basename(md_sys_dir)))
            merge_files_to_one(out_mlmd_list, mvm_save_file)
            mvm_list.append(mvm_save_file)
        return mvm_list
    
    