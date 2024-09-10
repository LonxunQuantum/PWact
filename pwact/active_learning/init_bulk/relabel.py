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

from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam
from pwact.active_learning.init_bulk.duplicate_scale import get_config_files_with_order

from pwact.utils.constant import PWMAT, INIT_BULK, TEMP_STRUCTURE, SLURM_OUT, DFT_STYLE
from pwact.active_learning.slurm.slurm import SlurmJob, Mission
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
    
from pwact.utils.file_operation import write_to_file, link_file, search_files, del_file_list_by_patten
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script
from pwact.data_format.configop import save_config, get_atom_type, load_config

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
       
    def make_scf_work(self):
        scf_paths = []
        use_dftb = False
        for init_config in self.init_configs:
            if init_config.scf is False:
                continue
            init_config_name = "init_config_{}".format(init_config.config_index)
            #1. read construtures from aimd dir

            #2. set relabel dir 
            #       read trajs from ./aimd/init_config_0/relax/0_aimd/
            #       make scf dir ./relabel/init_config_0/relax/0_aimd/10-scf/files
            traj_list = search_files(os.path.join(self.aimd_dir, init_config_name), "*/*aimd")
            for traj_dir in traj_list:
                scf_dir = os.path.join(self.scf_dir, init_config_name, \
                    os.path.basename(os.path.dirname(traj_dir)),\
                    os.path.basename(traj_dir))

                traj_file_name = DFT_STYLE.get_aimd_config(self.resource.dft_style)

                scf_lsit = self.make_scf_file(
                    scf_dir      =scf_dir, 
                    traj_file    =os.path.join(traj_dir, traj_file_name), 
                    traj_format  =DFT_STYLE.get_format_by_postfix(traj_file_name), 
                    interval     = self.input_param.interval,
                    target_format=DFT_STYLE.get_pwdata_format(self.input_param.scf_style, is_cp2k_coord=True),
                    input_file   =init_config.scf_input_file,
                    kspacing     =init_config.scf_kspacing, 
                    flag_symm    =init_config.scf_flag_symm,
                    is_dftb      = False,
                    in_skf       =None)

                scf_paths.extend(scf_lsit)
        # make slurm script and slurm job
        self.make_scf_slurm_job_files(scf_paths, use_dftb)
    
    def check_work_done(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(INIT_BULK.scf_job), \
            tag_patten="*-{}".format(INIT_BULK.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        return slurm_done
            
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

    '''
    description: 
        input_file is scf input control file, for vasp is incar, for pwmat is etot.input
    return {*}
    author: wuxingxing
    '''    
    def make_scf_file(self, scf_dir:str, traj_file:str, traj_format:str, interval:int, target_format:str, \
                input_file:str, kspacing:float=None, flag_symm:int=None, is_dftb:bool=False, in_skf:str=None):
        config = load_config(format=traj_format, config=traj_file)
        index_list = list(range(0, len(config), interval))
        scf_lsit = []
        for index in index_list:
            save_dir = os.path.join(scf_dir, "{}-{}".format(index, INIT_BULK.scf))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            #1. set config file
            target_config = save_config(config=config[index], 
                                        input_format=traj_format,# or None, the same
                                        wrap = False, 
                                        direct = True, 
                                        sort = True, 
                                        save_format=target_format, 
                                        save_path=save_dir, 
                                        save_name=DFT_STYLE.get_normal_config(self.input_param.scf_style))

            atom_type_list, _ = get_atom_type(config[index])
            #2. set pseudo files
            # if not is_dftb:
            pseudo_names = link_pseudo_by_atom(
                pseudo_list     = self.input_param.dft_input.scf_pseudo, 
                target_dir      = save_dir, 
                atom_order      = atom_type_list, 
                dft_style       = self.resource.scf_style,
                basis_set_file  =self.input_param.dft_input.basis_set_file,
                potential_file  =self.input_param.dft_input.potential_file
                )
            # else:
            #     # link in.skf path to aimd dir
            #     pseudo_names = []
            #     target_dir = os.path.join(aimd_dir, PWMAT.in_skf)
            #     link_file(in_skf, target_dir)
            #3. make dft input file
            set_input_script(
                input_file=input_file,
                config=target_config,
                dft_style=self.resource.scf_style,
                kspacing=kspacing, 
                flag_symm=flag_symm, 
                save_dir = save_dir,
                pseudo_names=pseudo_names,
                basis_set_file_name=self.input_param.dft_input.basis_set_file,# these for cp2k
                potential_file_name=self.input_param.dft_input.potential_file
            )
            scf_lsit.append(save_dir)
        return scf_lsit

    def make_scf_slurm_job_files(self, scf_dir_list:list[str],use_dftb: bool=False):
        del_file_list_by_patten(self.scf_dir, "*{}".format(INIT_BULK.scf_job))
        group_list = split_job_for_group(self.resource.scf_resource.group_size, scf_dir_list, self.resource.scf_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "scf{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.scf_tag)
            tag = os.path.join(self.scf_dir, tag_name)
            run_cmd = self.resource.scf_resource.command
            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.scf_resource.gpu_per_node, 
                number_node = self.resource.scf_resource.number_node, 
                cpu_per_node = self.resource.scf_resource.cpu_per_node,
                queue_name = self.resource.scf_resource.queue_name,
                custom_flags = self.resource.scf_resource.custom_flags,
                env_script = self.resource.scf_resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.scf_tag, 
                task_tag_faild = INIT_BULK.scf_tag_failed,
                parallel_num=self.resource.scf_resource.parallel_num,
                check_type=self.resource.scf_style
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.scf_job)
            slurm_job_file =  os.path.join(self.scf_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def do_post_process(self):
        if os.path.exists(self.scf_dir):
            link_file(self.scf_dir, self.real_scf_dir)
