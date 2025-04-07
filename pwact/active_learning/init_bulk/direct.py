"""
    do md with bigmodel
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
    or
    ------------/BIGMODEL
    -----------------/*_init_config/*_config_dir the config file from pertub or *_init_.config
    -------------------------------/bigmodel_md_script.py, POSCAR, ...
    ------------/DIRECT
    -----------------/*_init_config/*_config_dir the config file from pertub or *_init_.config
    -------------------------------/direct_script.py, POSCAR, ...

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
    
from pwact.utils.file_operation import write_to_file, link_file, del_file_list_by_patten, copy_file
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script
from pwact.data_format.configop import save_config, get_atom_type

class DIRECT(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resource = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
        self.super_cell_scale_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale)
        self.pertub_dir = os.path.join(self.input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
        self.bigmodel_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.bigmodel)
        self.real_bigmodel_dir = os.path.join(self.input_param.root_dir, INIT_BULK.bigmodel)
        self.direct_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.direct)
        self.real_direct_dir = os.path.join(self.input_param.root_dir, INIT_BULK.direct)
  
    def make_bigmodel_work(self):
        bigmodel_paths = []
        for init_config in self.init_configs:
            if init_config.bigmodel is False:
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
                bigmodel_dir = os.path.join(self.bigmodel_dir, init_config_name, tmp_config_dir, "{}_{}".format(index, INIT_BULK.bigmodel))
                if not os.path.exists(bigmodel_dir):
                    os.makedirs(bigmodel_dir)
                
                self.make_bigmodel_file(
                    bigmodel_dir=bigmodel_dir, 
                    config_file=config, 
                    conifg_format=DFT_STYLE.get_format_by_postfix(os.path.basename(config)), 
                    input_file=init_config.bigmodel_input_file)

                bigmodel_paths.append(bigmodel_dir)
        # make slurm script and slurm job
        self.make_bigmodel_slurm_job_files(bigmodel_paths)
    
    def check_work_done(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.bigmodel_dir, \
            job_patten="*-{}".format(INIT_BULK.bigmodel_job), \
            tag_patten="*-{}".format(INIT_BULK.bigmodel_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        return slurm_done
            
    def do_bigmodel_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.bigmodel_dir, \
            job_patten="*-{}".format(INIT_BULK.bigmodel_job), \
            tag_patten="*-{}".format(INIT_BULK.bigmodel_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these bigmodel Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), INIT_BULK.bigmodel_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()
                # mission.move_slurm_log_to_slurm_work_dir()

    '''
    description: 
        input_file is aimd input control file, for vasp is incar, for pwmat is etot.input
    return {*}
    author: wuxingxing
    '''    
    def make_bigmodel_file(
            bigmodel_dir=str, 
            config_file=str, 
            conifg_format=str, 
            input_file=str):
        #1. set config file, cvt to poscar format
        target_config = save_config(config=config_file, 
                            input_format=conifg_format,
                            wrap = False, 
                            direct = True, 
                            sort = True, 
                            save_format=DFT_STYLE.poscar, 
                            save_path=bigmodel_dir, 
                            save_name=DFT_STYLE.get_normal_config(DFT_STYLE.vasp)
                            )

        atom_type_list, _ = get_atom_type(config_file, conifg_format)
        #2. copy script file
        copy_file(input_file, os.path.join(bigmodel_dir, os.path.basename(input_file)))

    def make_bigmodel_slurm_job_files(self, bigmodel_dir_list:list[str]):
        del_file_list_by_patten(self.bigmodel_dir, "*{}".format(INIT_BULK.bigmodel_job))
        group_list = split_job_for_group(self.resource.explore_resource.group_size, bigmodel_dir_list, self.resource.explore_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "bigmodel{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.bigmodel_tag)
            tag = os.path.join(self.bigmodel_dir, tag_name)
            run_cmd = self.resource.explore_resource.command
            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.explore_resource.gpu_per_node, 
                number_node = self.resource.explore_resource.number_node, 
                cpu_per_node = self.resource.explore_resource.cpu_per_node,
                queue_name = self.resource.explore_resource.queue_name,
                custom_flags = self.resource.explore_resource.custom_flags,
                env_script = self.resource.explore_resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.bigmodel_tag, 
                task_tag_faild = INIT_BULK.bigmodel_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=self.resource.dft_style
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.bigmodel_job)
            slurm_job_file =  os.path.join(self.bigmodel_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def do_post_process(self):
        if os.path.exists(self.bigmodel_dir):
            link_file(self.bigmodel_dir, self.real_bigmodel_dir)
