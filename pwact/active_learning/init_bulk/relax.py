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
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam, Stage

from pwact.active_learning.slurm.slurm import SlurmJob, Mission
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
    
from pwact.utils.constant import PWMAT, INIT_BULK, TEMP_STRUCTURE, SLURM_OUT, DFT_STYLE, PWDATA

from pwact.utils.file_operation import write_to_file, link_file, search_files, mv_file, del_file, copy_file, del_file_list_by_patten
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script
from pwact.data_format.configop import save_config, get_atom_type

class Relax(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resource = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
        self.relax_real_dir = os.path.join(self.input_param.root_dir, INIT_BULK.relax)
        
    def check_work_done(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.relax_dir, \
        job_patten="*-{}".format(INIT_BULK.relax_job), \
        tag_patten="*-{}".format(INIT_BULK.relax_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        return slurm_done
    
    def make_relax_work(self):
        relax_paths = []
        for init_config in self.init_configs:
            if init_config.relax:
                init_config_name = "{}_{}".format(INIT_BULK.init_config, init_config.config_index)
                relax_path = os.path.join(self.relax_dir, init_config_name)
                if not os.path.exists(relax_path):
                    os.makedirs(relax_path)
                self.make_relax_file(relax_path, init_config=init_config)
                relax_paths.append(relax_path)
        # make slurm script and slurm job
        self.make_relax_slurm_job_files(relax_paths)
    
    def do_relax_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.relax_dir, \
            job_patten="*-{}".format(INIT_BULK.relax_job), \
            tag_patten="*-{}".format(INIT_BULK.relax_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False # len(slurm_remain) > 0 exist slurm jobs need to do
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these relax Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), INIT_BULK.relax_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.dft_out)
                # mission.move_slurm_log_to_slurm_work_dir()
                
    def make_relax_file(self, relax_path, init_config:Stage):
        #1. set config file
        target_config = save_config(config=init_config.config_file,
                                    input_format=init_config.format,
                                    wrap = False, 
                                    direct = True, 
                                    sort = True, 
                                    save_format=DFT_STYLE.get_pwdata_format(dft_style=self.input_param.dft_style, is_cp2k_coord=True), 
                                    save_path=relax_path, 
                                    save_name=DFT_STYLE.get_normal_config(self.input_param.dft_style))

        atom_type_list, _ = get_atom_type(init_config.config_file, init_config.format)
        #2. set pseudo files
        if not init_config.use_dftb:
            pseudo_names = link_pseudo_by_atom(
                pseudo_list     = self.input_param.dft_input.pseudo, 
                target_dir      = relax_path, 
                atom_order      = atom_type_list, 
                dft_style       = self.input_param.dft_style,
                basis_set_file  =self.input_param.dft_input.basis_set_file,
                potential_file  =self.input_param.dft_input.potential_file)
        else:
            # link in.skf path to aimd dir
            target_dir = os.path.join(relax_path, PWMAT.in_skf)
            if self.input_param.dft_input.in_skf is not None:
                link_file(self.input_param.dft_input.in_skf, target_dir)
            pseudo_names = []
        #3. make input file, for vasp is INCAR, for PWMAT is etot.input, for cp2k is inp file
        set_input_script(
            input_file=init_config.relax_input_file,
            config=target_config,
            kspacing=init_config.relax_kspacing, 
            flag_symm=init_config.relax_flag_symm, 
            dft_style=self.input_param.dft_style,
            save_dir=relax_path,
            pseudo_names=pseudo_names,
            basis_set_file_name=self.input_param.dft_input.basis_set_file,# these for cp2k
            potential_file_name=self.input_param.dft_input.potential_file,
            # xc_functional=self.input_param.dft_input.xc_functional,
            # potential=self.input_param.dft_input.potential,
            # basis_set=self.input_param.dft_input.basis_set
        )

    def make_relax_slurm_job_files(self, relax_sub_list:list[str]):
        del_file_list_by_patten(self.relax_dir, "*{}".format(INIT_BULK.relax_job))
        group_list = split_job_for_group(self.resource.dft_resource.group_size, relax_sub_list, self.resource.dft_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "relax{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.relax_tag)
            tag = os.path.join(self.relax_dir, tag_name)
            # if self.resource.dft_resource.gpu_per_node > 0:
            #     run_cmd = "mpirun -np {} PWmat".format(self.resource.dft_resource.gpu_per_node)
            # else:
            #     raise Exception("ERROR! the cpu version of pwmat not support yet!")
            run_cmd = self.resource.dft_resource.command
            group_slurm_script = set_slurm_script_content(
                number_node = self.resource.dft_resource.number_node, 
                gpu_per_node=self.resource.dft_resource.gpu_per_node, 
                cpu_per_node = self.resource.dft_resource.cpu_per_node,
                queue_name = self.resource.dft_resource.queue_name,
                custom_flags = self.resource.dft_resource.custom_flags,
                env_script = self.resource.dft_resource.env_script,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.relax_tag, 
                task_tag_faild = INIT_BULK.relax_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=self.resource.dft_style
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.relax_job)
            slurm_job_file = os.path.join(self.relax_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    '''
    description: 
     after relax calculate:
        for PWmat: change the 'final.config' to relaxed.config
        for VASP: change the 'CONTCAR' to  relaxed.poscar
        for CP2k: change the 'dft.log' to relaxed.poscar
        
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def do_post_process(self):
        # no use thie code
        # 1. change the name of relaxed config to relaxed.config for pwmat or relaxed.poscar to vasp
        final_configs = search_files(self.relax_dir, "*/{}".format(DFT_STYLE.get_relaxed_original_name(self.resource.dft_style)))
        for final_config in final_configs:
            target_config = os.path.join(os.path.dirname(final_config), DFT_STYLE.get_relaxed_config(self.resource.dft_style))
            if self.resource.dft_style == DFT_STYLE.cp2k:
                save_config(config=final_config, #convert cp2k scf to poscar format
                        input_format=PWDATA.cp2k_scf, 
                        direct=False, 
                        sort=False, 
                        save_path=os.path.dirname(target_config), 
                        save_name=os.path.basename(target_config),
                        save_format=PWDATA.vasp_poscar)
            else:# rename final.config to relaxed.config, or contcar to relaxed.poscar
                copy_file(final_config, target_config)
                
        # 2. link relax_dir to relax_real_dir
        if not os.path.exists(self.relax_dir):
            link_file(self.relax_dir, self.relax_real_dir)

    def delete_nouse_files(self):
        #1. mv init_config_* to real_dir
        mv_file(self.relax_dir, self.relax_real_dir)
        #2. delete tag and slurm files
        tag_list  = search_files(self.relax_real_dir,  template="*-tag-*")
        slurm_log = search_files(self.relax_real_dir,  template="*slurm-*.out")
        del_file(tag_list)
        del_file(slurm_log)
        
    