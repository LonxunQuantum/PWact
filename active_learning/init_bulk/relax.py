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
from active_learning.user_input.resource import Resource
from active_learning.user_input.init_bulk_input import InitBulkParam, Stage

from active_learning.slurm import SlurmJob, Mission
from utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
    
from utils.constant import PWMAT, INIT_BULK, TEMP_STRUCTURE, SLURM_OUT, DFT_STYLE

from utils.file_operation import write_to_file, link_file, search_files, mv_file, del_file, copy_file
from utils.app_lib.pwmat import set_etot_input_by_file
from utils.app_lib.common import link_pseudo_by_atom, get_atom_type, link_structure, set_input_script

class Relax(object):
    def __init__(self, resource: Resource, input_param:InitBulkParam):
        self.resource = resource
        self.input_param = input_param
        self.init_configs = self.input_param.sys_config
        self.relax_dir = os.path.join(self.input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
        self.relax_real_dir = os.path.join(self.input_param.root_dir, INIT_BULK.relax)
        
    def check_work_done(self):
        slurm_remain, slurm_done = get_slurm_job_run_info(self.relax_dir, \
        job_patten="*-{}".format(INIT_BULK.relax_job), \
        tag_patten="*-{}".format(INIT_BULK.relax_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        return slurm_done
    
    def make_relax_work(self):
        relax_paths = []
        for init_config in self.init_configs:
            if init_config.relax:
                init_config_name = "{}_{}".format(INIT_BULK.init_config, init_config.config_index)
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
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Doing these relax Jobs:\n")
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
                
    def make_relax_file(self, relax_path, init_conifg:Stage):
        #1. link config file
        target_config = link_structure(source_config = init_conifg.config, 
                                        config_format=init_conifg.format,
                                        target_dir = relax_path,
                                        dft_style=self.input_param.dft_style)
        atom_type_list, _ = get_atom_type(target_config, self.input_param.dft_style)
        #2. set pseudo files
        link_pseudo_by_atom(self.input_param.dft_input.pseudo, relax_path, atom_type_list, self.input_param.dft_style)
        #3. make input file, for vasp is INCAR, for PWMAT is etot.input
        set_input_script(
            input_file=init_conifg.relax_input_file,
            config=target_config,
            kspacing=init_conifg.relax_kspacing, 
            flag_symm=init_conifg.relax_flag_symm, 
            resource_node=[self.resource.dft_resource.number_node, self.resource.dft_resource.gpu_per_node],
            dft_style=self.input_param.dft_style,
            save_dir = relax_path
        )

    def make_relax_slurm_job_files(self, relax_sub_list:list[str]):
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
                source_list = self.resource.dft_resource.source_list,
                module_list = self.resource.dft_resource.module_list,
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

    def do_post_process(self):
        # no use thie code
        # 1. change the name of relaxed config to relaxed.config for pwmat or relaxed.poscar to vasp
        final_configs = search_files(self.relax_dir, "*/{}".format(INIT_BULK.get_relaxed_original_name(self.resource.dft_style)))
        for final_config in final_configs:
            target_config = os.path.join(os.path.dirname(final_config), INIT_BULK.get_relaxed_config(self.resource.dft_style))
            copy_file(final_config, target_config)
                
        # 2. link relax_dir to relax_real_dir
        if not os.path.exists(self.relax_real_dir):
            link_file(self.relax_dir, self.relax_real_dir)

    def delete_nouse_files(self):
        #1. mv init_config_* to real_dir
        mv_file(self.relax_dir, self.relax_real_dir)
        #2. delete tag and slurm files
        tag_list  = search_files(self.relax_real_dir,  template="*-tag-*")
        slurm_log = search_files(self.relax_real_dir,  template="*slurm-*.out")
        del_file(tag_list)
        del_file(slurm_log)
        
    