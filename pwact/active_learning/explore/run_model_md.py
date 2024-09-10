"""
    dir of explore:
    iter.0000/explore/md:
    --------------------/*-md.job file
    --------------------/*-md.tag.success file
    --------------------/md.000.sys.000/ dir
    -----------------------------------/md.000.sys.000.t.000.p.000 or md.000.sys.000.t.000 dir
    -------------------------------------------------------------/md files: lmp.config, in.lammps, forcefile files, model_devi file
    -------------------------------------------------------------/trajs
    iter.0000/explore/select:
    -------------------/summary.txt
    -------------------/accurate.txt candidate.txt failed.txt candidate_del.txt
    the content of candidate.txt is:
            traj_file_path1 index1
            traj_file_path2 index2
            ...
"""
from pwact.active_learning.slurm.slurm import Mission, SlurmJob, scancle_job
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
from pwact.active_learning.explore.select_image import select_image
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.iter_input import InputParam, MdDetail
from pwact.utils.constant import AL_STRUCTURE, TEMP_STRUCTURE, EXPLORE_FILE_STRUCTURE, TRAIN_FILE_STRUCTUR, \
        FORCEFILED, ENSEMBLE, LAMMPS, LAMMPS_CMD, UNCERTAINTY, DFT_STYLE, SLURM_OUT, SLURM_JOB_TYPE, PWDATA, MODEL_TYPE

from pwact.utils.format_input_output import get_iter_from_iter_name, get_sub_md_sys_template_name,\
    make_md_sys_name, get_md_sys_template_name, make_temp_press_name, make_temp_name, make_train_name
from pwact.utils.file_operation import write_to_file, add_postfix_dir, link_file, read_data, search_files, copy_dir, copy_file, del_file, del_dir, del_file_list, del_file_list_by_patten, mv_file
from pwact.utils.app_lib.lammps import make_lammps_input
from pwact.data_format.configop import save_config, get_atom_type

import os
import glob
import pandas as pd
"""
md_dir:
  a. pwmat+dpkf run md ->MOVEMENT
md_dpkf_dir:
  b. step a.MOVEMENT add Atomic-Energy block ->MOVEMENT
md_traj_dir:
  c. step a.MOVEMENT add Atomic-Energy block ->seperate MOVEMENT to atom.configs
kpu_dir:
  d. step b. calculate KPU of each image(MD)
  f. step d. select cadidate set by limited Delta0 and Delta1
"""
class Explore(object):
    @staticmethod
    def kill_job(root_dir:str, itername:str):
        explore_dir = os.path.join(root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.explore)
        md_dir = os.path.join(explore_dir, EXPLORE_FILE_STRUCTURE.md)
        scancle_job(md_dir)

    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.iter = get_iter_from_iter_name(self.itername)
        self.resource = resource
        self.input_param = input_param
        # self.sys_paths = self.input_param.explore.sys_configs
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, self.itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.train)
        self.real_train_dir = os.path.join(self.input_param.root_dir, self.itername, AL_STRUCTURE.train)
        
        # md work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.kpu_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.kpu) # for kpu calculate
       
        self.real_explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.real_md_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.real_select_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.real_kpu_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.kpu) # for kpu calculate

    def back_explore(self):
        slurm_remain, slurm_success = get_slurm_job_run_info(self.real_md_dir, \
        job_patten="*-{}".format(EXPLORE_FILE_STRUCTURE.md_job), \
        tag_patten="*-{}".format(EXPLORE_FILE_STRUCTURE.md_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done:
            # bk and do new job
            target_bk_file = add_postfix_dir(self.real_explore_dir, postfix_str="bk")
            mv_file(self.real_explore_dir, target_bk_file)
            # if the temp_work_dir/explore exists, delete the train dir
            if os.path.exists(self.explore_dir):
                del_dir(self.explore_dir)

    def make_md_work(self):
        md_work_list = []
        for md_index, md in enumerate(self.md_job):
            for sys_index in md.sys_idx:
                char_len = 3 if len(md.sys_idx) < 1000 else len(str(len(md.sys_idx)))
                md_sys_name = make_md_sys_name(md_index, sys_index, char_len)
                md_sys_dir = os.path.join(self.md_dir, md_sys_name)
                if not os.path.exists(md_sys_dir):
                    os.makedirs(md_sys_dir)
                for temp_index, temp in enumerate(md.temp_list):
                    if ENSEMBLE.nvt in md.ensemble:#for nvt ensemble
                        temp_name = make_temp_name(md_index, sys_index, temp_index, char_len)
                        temp_dir = os.path.join(md_sys_dir, temp_name)
                        # mkdir: md.000.sys.000/md.000.sys.000.t.000
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        self.set_md_files(len(md_work_list), temp_dir, sys_index, temp_index, None, md)
                        md_work_list.append(temp_dir)
                    elif ENSEMBLE.npt in md.ensemble: # for npt ensemble
                        for press_index, press in enumerate(md.press_list):
                            temp_press_name = make_temp_press_name(md_index, sys_index, temp_index, press_index, char_len)
                            temp_press_dir = os.path.join(md_sys_dir, temp_press_name)
                            # mkdir: md.000.sys.000/md.000.sys.000.p.000.t.000
                            if not os.path.exists(temp_press_dir):
                                os.makedirs(temp_press_dir)
                            self.set_md_files(len(md_work_list), temp_press_dir, sys_index, temp_index, press_index, md)
                            md_work_list.append(temp_press_dir)
                           
        self.make_md_slurm_jobs(md_work_list)
             
    def make_md_slurm_jobs(self, md_work_list:list[str]):
        # delete old job file
        del_file_list_by_patten(self.md_dir, "*{}".format(EXPLORE_FILE_STRUCTURE.md_job))
        group_list = split_job_for_group(self.resource.explore_resource.group_size, md_work_list, self.resource.explore_resource.parallel_num)
        for g_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            jobname = "md{}".format(g_index)
            tag_name = "{}-{}".format(g_index, EXPLORE_FILE_STRUCTURE.md_tag)
            tag = os.path.join(self.md_dir, tag_name)

            # if self.resource.explore_resource.gpu_per_node > 0:
            #     if self.input_param.strategy.uncertainty.upper() == UNCERTAINTY.committee.upper():
            #         gpu_per_node = 1
            #         cpu_per_node = 1
            #         run_cmd = "mpirun -np {} {} -in {} > {} ".format(1, LAMMPS_CMD.lmp_mpi_gpu, LAMMPS.input_lammps, SLURM_OUT.md_out)
            #     else:
            #         cpu_per_node = self.resource.explore_resource.gpu_per_node
            #         gpu_per_node = self.resource.explore_resource.gpu_per_node
            #         run_cmd = "mpirun -np {} {} -in {} > {} ".format(self.resource.explore_resource.gpu_per_node, LAMMPS_CMD.lmp_mpi_gpu, LAMMPS.input_lammps, SLURM_OUT.md_out)
            # else:
            #     cpu_per_node = self.resource.explore_resource.cpu_per_node
            #     run_cmd = "mpirun -np {} {} -in {} > {} ".format(self.resource.explore_resource.cpu_per_node, LAMMPS_CMD.lmp_mpi, LAMMPS.input_lammps, SLURM_OUT.md_out)

            run_cmd = self.resource.explore_resource.command

            group_slurm_script = set_slurm_script_content(
                            gpu_per_node=self.resource.explore_resource.gpu_per_node, 
                            number_node = self.resource.explore_resource.number_node, #1
                            cpu_per_node = self.resource.explore_resource.cpu_per_node,
                            queue_name = self.resource.explore_resource.queue_name,
                            custom_flags = self.resource.explore_resource.custom_flags,
                            env_script = self.resource.explore_resource.env_script,
                            job_name = jobname,
                            run_cmd_template = run_cmd,
                            group = group,
                            job_tag = tag,
                            task_tag = EXPLORE_FILE_STRUCTURE.md_tag,
                            task_tag_faild = EXPLORE_FILE_STRUCTURE.md_tag_faild,
                            parallel_num=self.resource.explore_resource.parallel_num,
                            check_type=None
                            )
            slurm_script_name = "{}-{}".format(g_index, EXPLORE_FILE_STRUCTURE.md_job)
            slurm_job_file = os.path.join(self.md_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    '''
    description: 
        waiting: if need set group size, make new script: work1 wait; work2 wait; ...
    param {*} self
    param {list} md_work_list
    return {*}
    author: wuxingxing
    '''    
    def do_md_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.md_dir, \
            job_patten="*-{}".format(EXPLORE_FILE_STRUCTURE.md_job), \
            tag_patten="*-{}".format(EXPLORE_FILE_STRUCTURE.md_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these MD Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), EXPLORE_FILE_STRUCTURE.md_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag, job_type=SLURM_JOB_TYPE.lammps)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.md_out)
                
                        
    def set_md_files(self, md_index:int, md_dir:str, sys_index:int, temp_index:int, press_index:int, md_detail:MdDetail):
        target_config = save_config(config=md_detail.config_file_list[sys_index],
                                    input_format=md_detail.config_file_format[sys_index],
                                    wrap = False, 
                                    direct = True, 
                                    sort = False, 
                                    save_format=PWDATA.lammps_lmp, 
                                    save_path=md_dir, 
                                    save_name=LAMMPS.lammps_sys_config)
        # import dpdata
        # _config = dpdata.System(md_detail.config_file_list[sys_index], fmt=md_detail.config_file_format[sys_index])
        # target_config = os.path.join(md_dir, LAMMPS.lammps_sys_config)
        # _config.to("lammps/lmp", target_config, frame_idx=0)
        #2. set forcefiled file
        md_model_paths = self.set_forcefiled_file(md_dir)
        
        #3. set lammps input file
        input_lammps_file = os.path.join(md_dir, LAMMPS.input_lammps)
        press=md_detail.press_list[press_index] if press_index is not None else None
        # get atom type
        atom_type_list, atomic_number_list = get_atom_type(md_detail.config_file_list[sys_index], md_detail.config_file_format[sys_index])
        atom_type_file = os.path.join(md_dir, LAMMPS.atom_type_file)
        write_to_file(atom_type_file, " ".join(atom_type_list), "w")
        restart_file = search_files(md_dir, "lmps.restart.*")
        restart = 1 if len(restart_file) > 0 else 0 
        lmp_input_content = make_lammps_input(
                        md_file=LAMMPS.lammps_sys_config, #save_file
                        md_type = self.input_param.strategy.md_type,
                        forcefiled = md_model_paths,
                        atom_type = atomic_number_list,
                        ensemble = md_detail.ensemble,
                        nsteps = md_detail.nsteps,
                        dt = md_detail.md_dt,
                        neigh_modify = md_detail.neigh_modify,
                        trj_freq = md_detail.trj_freq,
                        mass = md_detail.mass,
                        temp = md_detail.temp_list[temp_index],
                        tau_t=md_detail.taut, # for fix
                        press=press,
                        tau_p=md_detail.taup if press is not None else None, # for fix    
                        boundary=True, #true is 'p p p', false is 'f f f'
                        merge_traj=md_detail.merge_traj,
                        restart = restart,
                        model_deviation_file = EXPLORE_FILE_STRUCTURE.model_devi
        )
        write_to_file(input_lammps_file, lmp_input_content, "w")
        if md_detail.merge_traj is False:
            traj_dir = os.path.join(md_dir, "traj")
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
        
    '''
    description: 
    param {*} self
    param {str} md_dir
    param {list} forcefile
    return {*}
    author: wuxingxing
    '''    
    def set_forcefiled_file(self, md_dir:str):
        model_name = ""
        md_model_paths = []
        if self.input_param.train.model_type == MODEL_TYPE.nep:
            model_name += "{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.nep_model_lmps)
        elif self.input_param.train.model_type == MODEL_TYPE.dp:
            if self.input_param.strategy.md_type == FORCEFILED.libtorch_lmps:
                model_name += "{}/{}".format(TRAIN_FILE_STRUCTUR.model_record, TRAIN_FILE_STRUCTUR.script_dp_name)
            elif self.input_param.strategy.md_type == FORCEFILED.fortran_lmps:
                if self.input_param.strategy.compress:
                    raise Exception("ERROR! The compress model does not support fortran lammps md! Please change the 'md_type' to 2!")
                else:
                    model_name += "{}/{}".format(TRAIN_FILE_STRUCTUR.fortran_dp, TRAIN_FILE_STRUCTUR.fortran_dp_name)

        for model_index in range(self.input_param.strategy.model_num):
            model_name_i = "{}/{}".format(make_train_name(model_index), model_name)
            source_model_path = os.path.join(self.real_train_dir, model_name_i)
            target_model_path = os.path.join(md_dir, "{}_{}".format(model_index, os.path.basename(source_model_path)))
            link_file(source_model_path, target_model_path)
            md_model_paths.append(target_model_path)
        return md_model_paths
        
    '''
    description: 
        1. copy the explore/md under temp_work_dir to iter*/explore
        2. if reserve traj is false, delete the trajs under iter*/explore/md/*/traj dir
           if use kpu, copy the kpu_model_devi.out to iter*/explore
           delte slurm*.out log files and lammps.log file
        4. copy the explore/select under temp_work_dir to iter*/explore
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def post_process_md(self):
        md_sys_dir_list = search_files(self.md_dir, get_md_sys_template_name())
        is_kpu = self.input_param.strategy.uncertainty.upper() == UNCERTAINTY.kpu.upper()
        for md_sys_dir in md_sys_dir_list:
            sub_md_sys_dir_list =search_files(md_sys_dir, get_md_sys_template_name())
            for sub_md_sys in sub_md_sys_dir_list:
                target_dir = sub_md_sys.replace(TEMP_STRUCTURE.tmp_run_iter_dir, "")
                copy_dir(sub_md_sys, target_dir)# delete trajs
                if self.input_param.reserve_md_traj and self.input_param.reserve_work:
                    pass
                else:
                    del_file_list([os.path.join(target_dir, EXPLORE_FILE_STRUCTURE.traj)])
                if is_kpu:
                    kpu_source_file = os.path.join(self.kpu_dir, os.path.basename(md_sys_dir), os.path.basename(sub_md_sys), EXPLORE_FILE_STRUCTURE.kpu_model_devi)
                    kpu_taget_file = os.path.join(target_dir, EXPLORE_FILE_STRUCTURE.kpu_model_devi)
                    if os.path.exists(kpu_source_file):
                        copy_file(kpu_source_file, kpu_taget_file)
                if self.input_param.reserve_work is False:#delete lammps.log
                    del_file(os.path.join(target_dir, LAMMPS.log_lammps))
                md_slurms = search_files(self.real_md_dir, "slurm-*") #delete slurm log files
                del_file_list(md_slurms)
                md_tag_slurm_scripts = search_files(self.md_dir, "*md.success")
                md_slurm_scripts = search_files(self.md_dir, "*md.job")
                for file in md_tag_slurm_scripts:
                    copy_file(file, os.path.join(self.real_md_dir, os.path.basename(file)))
                for file in md_slurm_scripts:
                    copy_file(file, os.path.join(self.real_md_dir, os.path.basename(file)))
        copy_dir(self.select_dir, self.real_select_dir)
    
    '''
    description: 
    select structure of system by model deviation (committee method)
    the candidate.csv columns is ["devi_force", "file_path", "config_index"]
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def select_image_by_committee(self):
        #1. get model_deviation file
        model_deviation_patten = "{}/{}".format(get_sub_md_sys_template_name(), EXPLORE_FILE_STRUCTURE.model_devi)
        model_devi_files = search_files(self.md_dir, model_deviation_patten)
        model_devi_files = sorted(model_devi_files)
        
        #2. for each file, read the model_deviation
        devi_pd = pd.DataFrame(columns=EXPLORE_FILE_STRUCTURE.devi_columns)
        for devi_file in model_devi_files:
            devi_force = read_data(devi_file, skiprows=0)
            tmp_pd = pd.DataFrame()
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] = devi_force[:, 1]
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[1]] = devi_force[:, 0]
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[2]] = os.path.dirname(devi_file)
            devi_pd = pd.concat([devi_pd, tmp_pd])
        devi_pd.reset_index(drop=True, inplace=True)
        devi_pd["config_index"].astype(int)
        #3. select images with lower and upper limitation
        summary_info, summary = select_image(save_dir=self.select_dir, 
                        devi_pd=devi_pd, 
                        lower=self.input_param.strategy.lower_model_deiv_f, 
                        higer=self.input_param.strategy.upper_model_deiv_f, 
                        max_select=self.input_param.strategy.max_select)
        print("Image select result:\n {}\n\n".format(summary_info))
        return summary

        
