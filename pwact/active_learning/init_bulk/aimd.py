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
    
from pwact.utils.file_operation import write_to_file, link_file, del_file_list_by_patten
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script
from pwact.data_format.configop import save_config, get_atom_type

class AIMD(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resource = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
        self.super_cell_scale_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale)
        self.pertub_dir = os.path.join(self.input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
        self.aimd_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.aimd)
        self.real_aimd_dir = os.path.join(self.input_param.root_dir, INIT_BULK.aimd)
        
    def make_aimd_work(self):
        aimd_paths = []
        use_dftb = False
        for init_config in self.init_configs:
            if init_config.aimd is False:
                continue
            init_config_name = "init_config_{}".format(init_config.config_index)
            # config_list, config_type = get_config_files_with_order(self.super_cell_scale_dir, self.relax_dir, init_config_name, init_config.config, self.pertub_dir)
            config_list, config_type = get_config_files_with_order(
            super_cell_scale_dir=self.super_cell_scale_dir,
            relax_dir=self.relax_dir,
            init_config_dirname=init_config_name, 
            init_config_path=init_config.config_file, 
            pertub_dir=self.pertub_dir,
            dft_style=self.resource.dft_style
            )

            for index, config in enumerate(config_list):
                if config_type == INIT_BULK.pertub:
                    tmp_config_dir = os.path.basename(os.path.basename(os.path.dirname(config)))
                elif config_type == INIT_BULK.super_cell or config_type == INIT_BULK.scale:
                    tmp_config_dir = os.path.basename(config).replace(DFT_STYLE.get_postfix(self.resource.dft_style), "")
                elif config_type == INIT_BULK.relax:
                    tmp_config_dir = INIT_BULK.relax
                else:
                    tmp_config_dir = INIT_BULK.init
                aimd_dir = os.path.join(self.aimd_dir, init_config_name, tmp_config_dir, "{}_{}".format(index, INIT_BULK.aimd))
                if not os.path.exists(aimd_dir):
                    os.makedirs(aimd_dir)
                self.make_aimd_file(
                    aimd_dir=aimd_dir, 
                    config_file=config, 
                    conifg_format=DFT_STYLE.get_format_by_postfix(os.path.basename(config)), 
                    target_format=DFT_STYLE.get_pwdata_format(self.resource.dft_style, is_cp2k_coord=True),
                    input_file=init_config.aimd_input_file, 
                    kspacing=init_config.aimd_kspacing, 
                    flag_symm=init_config.aimd_flag_symm,
                    is_dftb = init_config.use_dftb,
                    in_skf=self.input_param.dft_input.in_skf)

                aimd_paths.append(aimd_dir)
                use_dftb = init_config.use_dftb
        # make slurm script and slurm job
        self.make_aimd_slurm_job_files(aimd_paths, use_dftb)
    
    def check_work_done(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.aimd_dir, \
            job_patten="*-{}".format(INIT_BULK.aimd_job), \
            tag_patten="*-{}".format(INIT_BULK.aimd_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        return slurm_done
            
    def do_aimd_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.aimd_dir, \
            job_patten="*-{}".format(INIT_BULK.aimd_job), \
            tag_patten="*-{}".format(INIT_BULK.aimd_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these aimd Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), INIT_BULK.aimd_tag)
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
        input_file is aimd input control file, for vasp is incar, for pwmat is etot.input
    return {*}
    author: wuxingxing
    '''    
    def make_aimd_file(self, aimd_dir:str, config_file:str, conifg_format:str, target_format:str, \
                input_file:str, kspacing:float=None, flag_symm:int=None, is_dftb:bool=False, in_skf:str=None):
        #1. set config file
        target_config = save_config(config=config_file, 
                                    input_format=conifg_format,
                                    wrap = False, 
                                    direct = True, 
                                    sort = True, 
                                    save_format=DFT_STYLE.get_pwdata_format(dft_style=self.resource.dft_style, is_cp2k_coord=True), 
                                    save_path=aimd_dir, 
                                    save_name=DFT_STYLE.get_normal_config(self.resource.dft_style))

        atom_type_list, _ = get_atom_type(config_file, conifg_format)
        #2. set pseudo files
        if not is_dftb:
            pseudo_names = link_pseudo_by_atom(
                pseudo_list     = self.input_param.dft_input.pseudo, 
                target_dir      = aimd_dir, 
                atom_order      = atom_type_list, 
                dft_style       = self.resource.dft_style,
                basis_set_file  =self.input_param.dft_input.basis_set_file,
                potential_file  =self.input_param.dft_input.potential_file
                )
        else:
            # link in.skf path to aimd dir
            pseudo_names = []
            target_dir = os.path.join(aimd_dir, PWMAT.in_skf)
            if in_skf is not None:
                link_file(in_skf, target_dir)
        #3. make dft input file
        set_input_script(
            input_file=input_file,
            config=target_config,
            dft_style=self.resource.dft_style,
            kspacing=kspacing, 
            flag_symm=flag_symm, 
            save_dir = aimd_dir,
            pseudo_names=pseudo_names,
            basis_set_file_name=self.input_param.dft_input.basis_set_file,# these for cp2k
            potential_file_name=self.input_param.dft_input.potential_file
        )

    def make_aimd_slurm_job_files(self, aimd_dir_list:list[str],use_dftb: bool=False):
        del_file_list_by_patten(self.aimd_dir, "*{}".format(INIT_BULK.aimd_job))
        group_list = split_job_for_group(self.resource.dft_resource.group_size, aimd_dir_list, self.resource.dft_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "aimd{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.aimd_tag)
            tag = os.path.join(self.aimd_dir, tag_name)
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
                task_tag = INIT_BULK.aimd_tag, 
                task_tag_faild = INIT_BULK.aimd_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=self.resource.dft_style
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.aimd_job)
            slurm_job_file =  os.path.join(self.aimd_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def do_post_process(self):
        if os.path.exists(self.aimd_dir):
            link_file(self.aimd_dir, self.real_aimd_dir)
