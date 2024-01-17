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
from active_learning.user_input.iter_input import SCFParam
from active_learning.user_input.init_bulk_input import InitBulkParam, Stage

from active_learning.slurm import SlurmJob, Mission
from utils.slurm_script import CHECK_TYPE,\
    get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
    
from utils.constant import PWMAT, INIT_BULK, TEMP_STRUCTURE

from utils.file_operation import write_to_file, link_file, search_files, mv_file, del_file
from utils.app_lib.pwmat import set_etot_input_by_file, get_atom_type_from_atom_config

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
                mission.all_job_finished()
                # mission.move_slurm_log_to_slurm_work_dir()
                
    def make_relax_file(self, relax_path, init_conifg:Stage):
        #1. link config file
        target_atom_config = os.path.join(relax_path, os.path.basename(init_conifg.config))
        link_file(init_conifg.config, target_atom_config)
        #2. make relax etot.input file
        # from atom.config get atom type
        atom_type_list, _ = get_atom_type_from_atom_config(init_conifg.config)
        pseudo_list = []
        for atom in atom_type_list:
            pseudo_atom_path = SCFParam.get_pseudo_by_atom_name(self.input_param.etot_input.pseudo, atom)
            pseduo_name = os.path.basename(pseudo_atom_path)
            link_file(pseudo_atom_path, os.path.join(relax_path, pseduo_name))
            pseudo_list.append(pseduo_name)
        #3. make etot.input file
        etot_script = set_etot_input_by_file(
            init_conifg.relax_etot_file, init_conifg.relax_kspacing, init_conifg.relax_flag_symm,\
                target_atom_config, [self.resource.scf_resource.number_node, self.resource.scf_resource.gpu_per_node])
        etot_input_file = os.path.join(relax_path, PWMAT.etot_input)
        write_to_file(etot_input_file, etot_script, "w")
        # if self.input_param.scf.etot_input_file is not None:
        #     etot_script = set_etot_input_by_file(self.input_param.scf.etot_input_file, target_atom_config, [self.resource.scf_resource.number_node, self.resource.scf_resource.gpu_per_node])
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


    def make_relax_slurm_job_files(self, relax_sub_list:list[str]):
        group_list = split_job_for_group(self.resource.scf_resource.group_size, relax_sub_list, self.resource.scf_resource.parallel_num)
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "relax{}".format(group_index)
            tag_name = "{}-{}".format(group_index, INIT_BULK.relax_tag)
            tag = os.path.join(self.relax_dir, tag_name)
            if self.resource.scf_resource.gpu_per_node > 0:
                run_cmd = "mpirun -np {} PWmat".format(self.resource.scf_resource.gpu_per_node)
            else:
                raise Exception("ERROR! the cpu version of pwmat not support yet!")
            group_slurm_script = set_slurm_script_content(gpu_per_node=self.resource.scf_resource.gpu_per_node, 
                number_node = self.resource.scf_resource.number_node, 
                cpu_per_node = self.resource.scf_resource.cpu_per_node,
                queue_name = self.resource.scf_resource.queue_name,
                custom_flags = self.resource.scf_resource.custom_flags,
                source_list = self.resource.scf_resource.source_list,
                module_list = self.resource.scf_resource.module_list,
                job_name = jobname,
                run_cmd_template = run_cmd,
                group = group,
                job_tag = tag,
                task_tag = INIT_BULK.relax_tag, 
                task_tag_faild = INIT_BULK.relax_tag_failed,
                parallel_num=self.resource.scf_resource.parallel_num,
                check_type=CHECK_TYPE.pwmat
                )
            slurm_script_name = "{}-{}".format(group_index, INIT_BULK.relax_job)
            slurm_job_file = os.path.join(self.relax_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    def do_post_process(self):
        # 1. link relax_dir to relax_real_dir
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
        
    