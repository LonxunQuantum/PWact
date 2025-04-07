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
import os
import glob
import json
import bisect
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam
from pwact.active_learning.init_bulk.duplicate_scale import get_config_files_with_order

from pwact.utils.constant import PWMAT, INIT_BULK, TEMP_STRUCTURE, SLURM_OUT, DFT_STYLE, PWDATA, VASP
from pwact.active_learning.slurm.slurm import SlurmJob, Mission
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
    
from pwact.utils.file_operation import write_to_file, link_file, del_dir, del_file_list_by_patten, get_random_nums
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script
from pwact.data_format.configop import extract_pwdata, save_config, get_atom_type, load_config

import pandas as pd
from pwdata import Config

# from pwact.utils.constant import DFT_TYPE, VASP, PWDATA, AL_STRUCTURE, TEMP_STRUCTURE,\
#     LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, LAMMPS, SLURM_OUT, DFT_STYLE, PWMAT, INIT_BULK
# from pwact.utils.file_operation import write_to_file, copy_file, copy_dir, search_files, mv_file, add_postfix_dir, del_dir, del_file_list_by_patten, link_file

class Relabel(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resource = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
        self.super_cell_scale_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale)
        self.pertub_dir = os.path.join(self.input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
        self.aimd_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.aimd)
        self.real_aimd_dir = os.path.join(self.input_param.root_dir, INIT_BULK.aimd)
        
        self.scf_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.scf)
        self.real_scf_dir = os.path.join(self.input_param.root_dir, INIT_BULK.scf)

        self.bigmodel_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.bigmodel)
        self.real_bigmodel_dir = os.path.join(self.input_param.root_dir, INIT_BULK.bigmodel)
        
        self.direct_dir = os.path.join(self.bigmodel_dir, INIT_BULK.direct)
        self.real_direct_dir = os.path.join(self.real_bigmodel_dir, INIT_BULK.direct)

    def check_work_done(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(INIT_BULK.scf_job), \
            tag_patten="*-{}".format(INIT_BULK.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        return slurm_done

    def make_scf_work(self):
        def find_position_binary(prefix_sum, N):
            idx = bisect.bisect_right(prefix_sum, N)
            if idx == 0:
                return 0
            elif idx >= len(prefix_sum):
                return (len(prefix_sum)-1)
            else:
                return idx

        def compute_prefix_sum(arr):
            prefix_sum = []
            current_sum = 0
            for num in arr:
                current_sum += num
                prefix_sum.append(current_sum)
            return prefix_sum
                            
        candidate = Config(data_path=os.path.join(self.direct_dir, INIT_BULK.direct_traj), format=PWDATA.extxyz)
        # from idx get config idx
        candidate_idx = json.load(open(os.path.join(self.direct_dir, INIT_BULK.candidate_idx)))
        candidate_idx_sum = compute_prefix_sum([candidate_idx[_]['num'] for _ in candidate_idx.keys()])
        _tmp = Config(data_path=os.path.join(self.direct_dir, INIT_BULK.direct_traj), format=PWDATA.extxyz)
        scf_dir_list = []
        if self.input_param.dft_input.scf_max_num is not None:
            random_list = get_random_nums(0, len(candidate.images), self.input_param.dft_input.scf_max_num, seed=2024)
        else:
            random_list = None
        for index, image in enumerate(candidate.images):
            if random_list is not None and index not in random_list:
                continue
            _idx = find_position_binary(candidate_idx_sum, index)
            config_idx = candidate_idx["{}".format(_idx)]['idx']
            scf_dir = os.path.join(self.scf_dir, "{}".format(index))
            if not os.path.exists(scf_dir):
                os.makedirs(scf_dir)
                
            _tmp.images = [image]
            _tmp.to(data_path=scf_dir, data_name=PWMAT.atom_config,
                         format=PWDATA.pwmat_config)
            self.make_scf_file(
                    scf_dir      =scf_dir, 
                    traj_file    =os.path.join(scf_dir, PWMAT.atom_config), 
                    traj_format  =PWDATA.pwmat_config, 
                    target_format=DFT_STYLE.get_pwdata_format(self.input_param.dft_style, is_cp2k_coord=True),
                    input_file   =self.init_configs[config_idx].scf_input_file,
                    kspacing     =self.init_configs[config_idx].scf_kspacing, 
                    flag_symm    =self.init_configs[config_idx].scf_flag_symm,
                    is_dftb      = False,
                    in_skf       =None)

            scf_dir_list.append(scf_dir)
        
        self.make_scf_slurm_job_files(scf_dir_list)

    def make_scf_slurm_job_files(self, scf_dir_list:list[str]):
        del_file_list_by_patten(self.scf_dir, "*{}".format(INIT_BULK.scf_job))
        group_list = split_job_for_group(self.resource.dft_resource.group_size, scf_dir_list, self.resource.dft_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "scf{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.scf_tag)
            tag = os.path.join(self.scf_dir, tag_name)
            run_cmd = self.resource.dft_resource.command
            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.dft_resource.gpu_per_node, 
                number_node = self.resource.dft_resource.number_node, 
                cpu_per_node = self.resource.dft_resource.cpu_per_node,
                queue_name = self.resource.dft_resource.queue_name,
                custom_flags = self.resource.dft_resource.custom_flags,
                env_script = self.resource.dft_resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.scf_tag, 
                task_tag_faild = INIT_BULK.scf_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=None
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.scf_job)
            slurm_job_file =  os.path.join(self.scf_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def do_scf_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(INIT_BULK.scf_job), \
            tag_patten="*-{}".format(INIT_BULK.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these relabel Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), INIT_BULK.scf_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.dft_out)
                # mission.move_slurm_log_to_slurm_work_dir()

    def make_scf_file(self, 
                    scf_dir, 
                    traj_file    , 
                    traj_format  , # the input is pwmat/config
                    target_format,
                    input_file   ,
                    kspacing     =None, 
                    flag_symm    =None,
                    is_dftb      =None,
                    in_skf       =None,
                    atom_names:list[str]=None):
        if DFT_STYLE.pwmat == self.resource.dft_style:
            target_config = traj_file
            pass
        else:
            if DFT_STYLE.vasp == self.resource.dft_style: # when do scf, the vasp input file name is 'POSCAR'
                save_name = VASP.poscar
            else:
                save_name="{}".format(DFT_STYLE.get_normal_config(self.resource.dft_style))# for cp2k this param will be set as coord.xzy
            target_config = save_config(config=traj_file,
                                        input_format=traj_format,
                                        wrap = False, 
                                        direct = True, 
                                        sort = True, 
                                        save_name = save_name,
                                        save_format=DFT_STYLE.get_pwdata_format(dft_style=self.resource.dft_style, is_cp2k_coord=True),
                                        save_path=scf_dir, 
                                        atom_names=atom_names)

        #2.
        atomic_name_list, atomic_number_list = get_atom_type(traj_file, traj_format)
        #1. set pseudo files
        pseudo_names = link_pseudo_by_atom(
                pseudo_list = self.input_param.dft_input.pseudo, 
                target_dir = scf_dir, 
                atom_order = atomic_name_list, 
                dft_style = self.resource.dft_style,
                basis_set_file  =self.input_param.dft_input.basis_set_file,
                potential_file  =self.input_param.dft_input.potential_file
                )
    
        #2. make etot.input file
        set_input_script(
            input_file=input_file,
            config=target_config,
            dft_style=self.resource.dft_style,
            kspacing=kspacing, 
            flag_symm=flag_symm, 
            save_dir = scf_dir,
            pseudo_names=pseudo_names,
            gaussian_base_param=self.input_param.dft_input.gaussian_base_param,# these for cp2k
            is_scf = True
        )

    def do_post_process(self):
        if os.path.exists(self.scf_dir):
            link_file(self.scf_dir, self.real_scf_dir)
