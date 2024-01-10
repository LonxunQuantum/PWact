"""
    realx dor strcuture
    --/work_path/
    ------------/relax
    -----------------/*_init_config/*_init_.config, etot.input, pseudo files, final.config, mvms_relax
    ------------/super_cell_scale
    -----------------/*_init_config/scale.config, from final.config or *_init_.config
    ------------/perturb
    --------------------/*_init_config/scale.config or final.config
    ----------------------------------/structures/*.config_perturb
    ------------/AIMD
    -----------------/*_init_config/*_config_dir the config file from pertub or *_init_.config
    -------------------------------/etot.input, atom.config, ...
    -----------/result
    -----------------/*_init_config/relax_mvm_*_init_config if need; aimd_mvm_*_init_config if need
    
"""
import os, sys
import numpy as np
import pandas as pd
from math import ceil

from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import SCFParam
from active_learning.user_input.init_bulk_input import InitBulkParam, Stage

from active_learning.slurm import SlurmJob, Mission, get_slurm_sbatch_cmd
from utils.slurm_script import CONDA_ENV, CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, get_slurm_job_run_info, \
    set_slurm_comm_basis, split_job_for_group, get_job_tag_check_string, set_slurm_script_content
    
from utils.constant import AL_STRUCTURE, LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, PWMAT, INIT_BULK, LAMMPSFILE

from utils.format_input_output import get_iter_from_iter_name, get_md_sys_template_name
from utils.file_operation import write_to_file, copy_file, link_file, merge_files_to_one
from utils.app_lib.pwmat import set_etot_input_by_file, make_pwmat_input_dict, get_atom_type_from_atom_config

class Relax(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resouce = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, INIT_BULK.relax)
        
    def make_relax_work(self):
        relax_paths = []
        for init_config in self.init_configs:
            if init_config.relax:
                init_config_name = "init_config_{}".format(init_config.config_index)
                relax_path = os.path.join(self.relax_dir, init_config_name)
                if not os.path.exists(relax_path):
                    os.makedirs(relax_path)
                self.make_relax_file(relax_path, init_conifg=init_config)
                relax_paths.append(relax_path)
        # make slurm script and slurm job
        self.make_relax_slurm_job_files(relax_paths)
    
    def do_relax_jobs(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.relax_dir, \
            job_patten="*-{}".format(INIT_BULK.relax_job), \
            tag_patten="*-{}".format(INIT_BULK.relax_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done == False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Doing these relax Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_cmd = get_slurm_sbatch_cmd(os.path.dirname(script_path), os.path.basename(script_path))
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), INIT_BULK.relax_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd, os.path.dirname(script_path))
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
                # mission.move_slurm_log_to_slurm_work_dir()
                
    def make_relax_file(self, relax_path, init_conifg:Stage):
        #1. link config file
        target_atom_config = os.path.join(relax_path, PWMAT.atom_config)
        link_file(init_conifg.config, target_atom_config)
        #2. make relax etot.input file
        # from atom.config get atom type
        atom_type_list = get_atom_type_from_atom_config(init_conifg.config)
        pseudo_list = []
        for atom in atom_type_list:
            pseudo_atom_path = SCFParam.get_pseudo_by_atom_name(self.input_param.scf.pseudo, atom)
            pseduo_name = os.path.basename(pseudo_atom_path)
            link_file(pseudo_atom_path, os.path.join(relax_path, pseduo_name))
            pseudo_list.append(pseduo_name)
        #3. make etot.input file
        etot_script = set_etot_input_by_file(self.input_param.scf.relax_etot_input_file, target_atom_config, [self.resouce.scf_resource.number_node, self.resouce.scf_resource.gpu_per_node])
        # if self.input_param.scf.etot_input_file is not None:
        #     etot_script = set_etot_input_by_file(self.input_param.scf.etot_input_file, target_atom_config, [self.resouce.scf_resource.number_node, self.resouce.scf_resource.gpu_per_node])
        # else:
        #     scfparam = self.input_param.scf
        #     etot_script = make_pwmat_input_dict(
        #     node1 = scfparam.node1,
        #     node2 = scfparam.node2,
        #     job_type = PWMAT.relax,
        #     pseudo_list=pseudo_list,
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
        etot_input_file = os.path.join(relax_path, PWMAT.etot_input)
        write_to_file(etot_input_file, etot_script, "w")

    def make_relax_slurm_job_files(self, relax_sub_list:list[str]):
        group_list = split_job_for_group(self.resouce.scf_resource.group_size, relax_sub_list, self.resouce.scf_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "relax{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.relax_tag)
            tag = os.path.join(self.relax_dir, tag_name)
            if self.resouce.scf_resource.gpu_per_node > 0:
                run_cmd = "mpirun -np {} PWmat".format(self.resouce.scf_resource.gpu_per_node, LAMMPSFILE.input_lammps)
            else:
                raise Exception("ERROR! the cpu version of pwmat not support yet!")
            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resouce.scf_resource.gpu_per_node, 
                number_node = self.resouce.scf_resource.number_node, 
                cpu_per_node = self.resouce.scf_resource.cpu_per_node,
                queue_name = self.resouce.scf_resource.queue_name,
                custom_flags = self.resouce.scf_resource.custom_flags,
                source_list = self.resouce.scf_resource.source_list,
                module_list = self.resouce.scf_resource.module_list,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.relax_tag, 
                task_tag_faild = INIT_BULK.relax_tag_failed,
                parallel_num=self.resouce.scf_resource.parallel_num
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.relax_job)
            slurm_job_file = os.path.join(self.relax_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")
    
    def set_relax_slurm_job_script(self,group:list[str], job_name:str, tag:str):
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
                    self.resouce.scf_resource.queue_name)
            mpirun_cmd_template = "mpirun -np {} PWmat".format(self.resouce.scf_resource.gpu_per_node)
                            
        script += set_slurm_comm_basis(self.resouce.scf_resource.custom_flags, \
            self.resouce.scf_resource.source_list, \
                self.resouce.scf_resource.module_list)
        
        # set conda env
        script += "\n"
        script += CONDA_ENV
        script += "\n"        
        script += "start=$(date +%s)\n"

        scf_cmd = ""
        job_id = 0
        job_tag_list = []
        while job_id < len(group):
            for i in range(self.resouce.pwmat_run_num):
                if group[job_id] == "NONE":
                    job_id += 1
                    continue
                scf_cmd += "{\n"
                scf_cmd += "cd {}\n".format(group[job_id])
                scf_cmd += "if [ ! -f {} ] ; then\n".format(INIT_BULK.relax_tag)
                scf_cmd += "    {}\n".format(mpirun_cmd_template)
                scf_cmd += "    if test $? -eq 0; then touch {}; else touch {}; fi\n".format(INIT_BULK.relax_tag, INIT_BULK.relax_tag_failed)
                scf_cmd += "fi\n"
                scf_cmd += "} &\n\n"
                job_tag_list.append(os.path.join(group[job_id], INIT_BULK.relax_tag))
                job_id += 1
            scf_cmd += "wait\n\n"
        
        script += scf_cmd
        
        right_script = ""
        right_script += "    end=$(date +%s)\n"
        right_script += "    take=$(( end - start ))\n"
        right_script += "    echo Time taken to execute commands is ${{take}} seconds > {}\n".format(tag)
        right_script += "    exit 0\n"   
        error_script  = "    exit 1\n"
        
        script += get_job_tag_check_string(job_tag_list, right_script, error_script)
        return script