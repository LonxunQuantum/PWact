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
import os
import glob
import pandas as pd

from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.iter_input import InputParam
from pwact.active_learning.slurm.slurm import SlurmJob, Mission, scancle_job

from pwact.utils.constant import DFT_TYPE, VASP, PWDATA, AL_STRUCTURE, TEMP_STRUCTURE,\
    LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, LAMMPS, SLURM_OUT, DFT_STYLE, PWMAT, INIT_BULK
    
from pwact.utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
from pwact.utils.format_input_output import get_iter_from_iter_name, get_md_sys_template_name
from pwact.utils.file_operation import write_to_file, copy_file, copy_dir, search_files, mv_file, add_postfix_dir, del_dir, del_file_list_by_patten, link_file
from pwact.utils.app_lib.common import link_pseudo_by_atom, set_input_script, check_convergence

from pwact.data_format.configop import extract_pwdata, save_config, get_atom_type
from pwdata import Config

class Labeling(object):
    @staticmethod
    def kill_job(root_dir:str, itername:str):
        label_dir = os.path.join(root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.labeling) 
        scf_dir = os.path.join(label_dir, LABEL_FILE_STRUCTURE.scf)
        scancle_job(scf_dir)

    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.iter = get_iter_from_iter_name(self.itername)
        self.resource = resource
        self.input_param = input_param
        
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        
        # train work dir
        self.train_dir = os.path.join(self.input_param.root_dir, self.itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.train)
        self.real_train_dir = os.path.join(self.input_param.root_dir, self.itername, AL_STRUCTURE.train)

        # md work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.explore)
        self.real_explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.direct_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.direct) 
        self.real_md_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.real_select_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.select)
        self.real_direct_dir = os.path.join(self.real_explore_dir, EXPLORE_FILE_STRUCTURE.direct)
        # labed work dir
        self.label_dir = os.path.join(self.input_param.root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.labeling) 
        self.scf_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.scf)
        self.result_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.result)

        self.real_label_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.labeling) 
        self.real_scf_dir = os.path.join(self.real_label_dir, LABEL_FILE_STRUCTURE.scf)
        self.real_result_dir = os.path.join(self.real_label_dir, LABEL_FILE_STRUCTURE.result)

        self.bigmodel_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.bigmodel)
        self.real_bigmodel_dir = os.path.join(self.real_label_dir, LABEL_FILE_STRUCTURE.bigmodel)

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
        candidate = pd.read_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
        # make scf work dir
        scf_dir_list = []
        for index, row in candidate.iterrows():
            config_index    = int(row["config_index"])
            sub_md_sys_path = row["file_path"]
            sub_md_sys_name = os.path.basename(sub_md_sys_path)
            md_sys_path     = os.path.dirname(sub_md_sys_path)
            md_sys_name     = os.path.basename(md_sys_path)
            scf_sub_md_sys_path = os.path.join(self.scf_dir, md_sys_name, sub_md_sys_name, "{}-{}".format(config_index, LABEL_FILE_STRUCTURE.scf))
            if not os.path.exists(scf_sub_md_sys_path):
                os.makedirs(scf_sub_md_sys_path)
            tarj_lmp = os.path.join(sub_md_sys_path, EXPLORE_FILE_STRUCTURE.traj, "{}{}".format(config_index, LAMMPS.traj_postfix))
            atom_names = None
            with open(os.path.join(sub_md_sys_path, LAMMPS.atom_type_file), 'r') as rf:
                line = rf.readline()
                atom_names = line.split()
            self.make_scf_file(scf_sub_md_sys_path, tarj_lmp, atom_names)
            scf_dir_list.append(scf_sub_md_sys_path)
        
        self.make_scf_slurm_job_files(scf_dir_list)

    def make_bigmodel_work(self):
        # copy from realdir/direct/select.xyz
        if self.input_param.strategy.direct:
            copy_file(os.path.join(self.real_direct_dir, EXPLORE_FILE_STRUCTURE.select_xyz), 
                os.path.join(self.bigmodel_dir, EXPLORE_FILE_STRUCTURE.select_xyz))
        else:
            # copy trajs to bigmodel_dir and cvt to xyz
            candidate = pd.read_csv(os.path.join(self.select_dir, EXPLORE_FILE_STRUCTURE.candidate))
            # make scf work dir
            image_list = None
            for index, row in candidate.iterrows():
                config_index    = int(row["config_index"])
                sub_md_sys_path = row["file_path"]
                atom_names = None
                with open(os.path.join(sub_md_sys_path, LAMMPS.atom_type_file), 'r') as rf:
                    line = rf.readline()
                    atom_names = line.split()
                if image_list is None:
                    image_list = Config(data_path=os.path.join(sub_md_sys_path, EXPLORE_FILE_STRUCTURE.traj, "{}{}".format(config_index, LAMMPS.traj_postfix)), 
                                        format=PWDATA.lammps_dump, atom_names=atom_names)
                else:
                    image_list.append(Config(data_path=os.path.join(sub_md_sys_path, EXPLORE_FILE_STRUCTURE.traj, "{}{}".format(config_index, LAMMPS.traj_postfix)), 
                                        format=PWDATA.lammps_dump, atom_names=atom_names))
            # cvt_lammps.dump to extxyz
            image_list.to(data_path=self.bigmodel_dir, format=PWDATA.extxyz, data_name="{}".format(EXPLORE_FILE_STRUCTURE.select_xyz))
        # copy bigmodelscript
        copy_file(self.input_param.scf.bigmodel_script, os.path.join(self.bigmodel_dir, os.path.basename(self.input_param.scf.bigmodel_script)))
        # make slrum file
        self.make_bigmodel_slurm_job_files([self.bigmodel_dir])

    def back_label(self):
        if self.input_param.scf.dft_style == DFT_STYLE.bigmodel:
            slurm_remain, slurm_success = get_slurm_job_run_info(self.real_bigmodel_dir, \
                job_patten="*-{}".format(LABEL_FILE_STRUCTURE.bigmodel_job), \
                tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.bigmodel_tag),
                for_back = True)
            slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False        
        else:
            slurm_remain, slurm_success = get_slurm_job_run_info(self.real_scf_dir, \
                job_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_job), \
                tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_tag))
            slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        
        if slurm_done:
            # bk and do new job
            target_bk_file = add_postfix_dir(self.real_label_dir, postfix_str="bk")
            mv_file(self.real_label_dir, target_bk_file)
            # if the temp_work_dir/label exists, delete the train dir
            if os.path.exists(self.label_dir):
                del_dir(self.label_dir)

    def do_scf_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_job), \
            tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run these SCF Jobs:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), LABEL_FILE_STRUCTURE.scf_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished(error_type=SLURM_OUT.dft_out)

    def do_bigmodel_jobs(self):
        mission = Mission()
        slurm_remain, slurm_success = get_slurm_job_run_info(self.bigmodel_dir, \
            job_patten="*-{}".format(LABEL_FILE_STRUCTURE.bigmodel_job), \
            tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.bigmodel_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_success) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Run bigModel Job:\n")
                print(slurm_remain)
                for i, script_path in enumerate(slurm_remain):
                    slurm_job = SlurmJob()
                    tag_name = "{}-{}".format(os.path.basename(script_path).split('-')[0].strip(), LABEL_FILE_STRUCTURE.bigmodel_tag)
                    tag = os.path.join(os.path.dirname(script_path),tag_name)
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(script_path)
                    mission.add_job(slurm_job)

            if len(mission.job_list) > 0:
                mission.commit_jobs()
                mission.check_running_job()
                mission.all_job_finished()

    def make_scf_file(self, scf_dir:str, tarj_lmp:str, atom_names:list[str]=None):
        config_index = os.path.basename(tarj_lmp).split('.')[0]
        if DFT_STYLE.vasp == self.resource.dft_style: # when do scf, the vasp input file name is 'POSCAR'
            save_name = VASP.poscar
        else:
            save_name="{}{}".format(config_index, DFT_STYLE.get_normal_config(self.resource.dft_style))# for cp2k this param will be set as coord.xzy
        target_config = save_config(config=tarj_lmp,
                                    input_format=PWDATA.lammps_dump,
                                    wrap = False, 
                                    direct = True, 
                                    sort = True, 
                                    save_name = save_name,
                                    save_format=DFT_STYLE.get_pwdata_format(dft_style=self.resource.dft_style, is_cp2k_coord=True),
                                    save_path=scf_dir, 
                                    atom_names=atom_names)

        #2.
        atom_type_list = atom_names
        #1. set pseudo files
        if not self.input_param.scf.use_dftb:
            pseudo_names = link_pseudo_by_atom(
                    pseudo_list = self.input_param.scf.pseudo, 
                    target_dir = scf_dir, 
                    atom_order = atom_type_list, 
                    dft_style = self.resource.dft_style,
                    basis_set_file  =self.input_param.scf.basis_set_file,
                    potential_file  =self.input_param.scf.potential_file
                    )
        else:
            # link in.skf path to aimd dir
            target_dir = os.path.join(scf_dir, PWMAT.in_skf)
            if self.input_param.scf.in_skf is not None:
                link_file(self.input_param.scf.in_skf, target_dir)
            pseudo_names = []
        #2. make etot.input file
        set_input_script(
            input_file=self.input_param.scf.scf_input_list[0].input_file,
            config=target_config,
            dft_style=self.resource.dft_style,
            kspacing=self.input_param.scf.scf_input_list[0].kspacing, 
            flag_symm=self.input_param.scf.scf_input_list[0].flag_symm, 
            save_dir=scf_dir,
            pseudo_names=pseudo_names,
            gaussian_base_param=self.input_param.scf.gaussian_base_param,# these for cp2k
            is_scf = True
        )
        
    def make_scf_slurm_job_files(self, scf_sub_list:list[str]):
        del_file_list_by_patten(self.scf_dir, "*{}".format(LABEL_FILE_STRUCTURE.scf_job))
        group_list = split_job_for_group(self.resource.dft_resource.group_size, scf_sub_list, self.resource.dft_resource.parallel_num)
        
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            
            jobname = "scf{}".format(group_index)
            tag_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.scf_tag)
            tag = os.path.join(self.scf_dir, tag_name)
            run_cmd = self.resource.dft_resource.command
            # if self.resource.dft_resource.gpu_per_node > 0:
            #     run_cmd = "mpirun -np {} PWmat > {}".format(self.resource.dft_resource.gpu_per_node, SLURM_OUT.md_out)
            # else:
            #     raise Exception("ERROR! the cpu version of pwmat not support yet!")
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
                task_tag = LABEL_FILE_STRUCTURE.scf_tag, 
                task_tag_faild = LABEL_FILE_STRUCTURE.scf_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=self.resource.dft_style
                )
            slurm_script_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.scf_job)
            slurm_job_file = os.path.join(self.scf_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")


    def make_bigmodel_slurm_job_files(self, scf_sub_list:list[str]):
        del_file_list_by_patten(self.bigmodel_dir, "*{}".format(LABEL_FILE_STRUCTURE.scf_job))
        group_list = split_job_for_group(1, scf_sub_list, 1)
        
        for group_index, group in enumerate(group_list):
            if group[0] == "NONE":
                continue
            
            jobname = "bigmodel{}".format(group_index)
            tag_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.bigmodel_tag)
            tag = os.path.join(self.bigmodel_dir, tag_name)
            run_cmd = self.resource.dft_resource.command
            # if self.resource.dft_resource.gpu_per_node > 0:
            #     run_cmd = "mpirun -np {} PWmat > {}".format(self.resource.dft_resource.gpu_per_node, SLURM_OUT.md_out)
            # else:
            #     raise Exception("ERROR! the cpu version of pwmat not support yet!")
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
                task_tag = LABEL_FILE_STRUCTURE.bigmodel_tag, 
                task_tag_faild = LABEL_FILE_STRUCTURE.bigmodel_tag_failed,
                parallel_num=self.resource.dft_resource.parallel_num,
                check_type=self.resource.dft_style
                )
            slurm_script_name = "{}-{}".format(group_index, LABEL_FILE_STRUCTURE.bigmodel_job)
            slurm_job_file = os.path.join(self.bigmodel_dir, slurm_script_name)
            write_to_file(slurm_job_file, group_slurm_script, "w")

    '''
    description: 
    collecte OUT.MLMD to mvm-
    param {*} self
    return {*}
    author: wuxingxing
    '''         
    def collect_scf_configs(self):
        md_sys_mlmd = []
        md_sys_dir_list = search_files(self.scf_dir, get_md_sys_template_name())
        for md_sys_dir in md_sys_dir_list:
            sub_md_sys_dir_list =search_files(md_sys_dir, get_md_sys_template_name())
            for sub_md_sys in sub_md_sys_dir_list:
                out_mlmd_list =search_files(sub_md_sys, "*-{}/{}".format(LABEL_FILE_STRUCTURE.scf, DFT_STYLE.get_scf_config(self.resource.dft_style)))
                # do a sorted?
                md_sys_mlmd.extend(out_mlmd_list)
        return md_sys_mlmd

    def get_aimd_list(self):
        aimd_list = search_files(self.scf_dir, "{}/{}".format(get_md_sys_template_name(), DFT_STYLE.get_aimd_config(self.resource.dft_style)))
        return sorted(aimd_list)
    
    '''
    description: 
        if reserve_work is True
            reserve all temp work dir
        else:
            if reserve_scf_files is True, copy scf and result files
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def do_post_labeling(self):
        # collect scf files
        scf_dirs = search_files(self.scf_dir, "{}/{}/*{}".format(get_md_sys_template_name(),get_md_sys_template_name(), LABEL_FILE_STRUCTURE.scf))
        for scf_dir in scf_dirs:
            if self.input_param.reserve_scf_files:
                target_scf_dir = scf_dir.replace(TEMP_STRUCTURE.tmp_run_iter_dir, "") 
                copy_dir(scf_dir, target_scf_dir)
            else:
                scf_files = os.listdir(scf_dir)
                for scf_file in scf_files:
                    scf_file_path = os.path.join(scf_dir, scf_file)
                    if scf_file.lower() in DFT_STYLE.get_scf_reserve_list(self.resource.dft_style) \
                        or "atom.config" in scf_file.lower() :# for the input natom.config
                        copy_file(scf_file_path, scf_file_path.replace(TEMP_STRUCTURE.tmp_run_iter_dir, ""))

        # scf files to pwdata format
        scf_configs = self.collect_scf_configs()
        cvg_files, uncvg_files, cvg_infos, cvg_detail_infos = check_convergence(scf_configs, self.resource.dft_style)
        if len(cvg_files) > 0:
            extract_pwdata(input_data_list=cvg_files,
                    intput_data_format =DFT_STYLE.get_format_by_postfix(os.path.basename(scf_configs[0])),
                    save_data_path =self.result_dir,
                    save_data_name = INIT_BULK.get_save_format(self.input_param.data_format),
                    save_data_format = self.input_param.data_format,
                    data_shuffle     =self.input_param.train.data_shuffle
            )
            # copy to main dir
            copy_dir(self.result_dir, self.real_result_dir)
        print(cvg_detail_infos)
        write_to_file(os.path.join(self.input_param.root_dir, EXPLORE_FILE_STRUCTURE.iter_select_file), cvg_detail_infos, mode='a')

    def do_post_bigmodel(self):
        # copy the bigmodel labeled.xyz to result
        if self.input_param.data_format == PWDATA.extxyz:
            copy_file(os.path.join(self.bigmodel_dir, LABEL_FILE_STRUCTURE.train_xyz), os.path.join(self.result_dir, LABEL_FILE_STRUCTURE.train_xyz))
        else:
            images = Config(data_path=os.path.join(self.bigmodel_dir, LABEL_FILE_STRUCTURE.train_xyz), format=PWDATA.extxyz)
            images.to(data_path=self.result_dir, format=PWDATA.pwmlff_npy)
        # copy bigmodel dir to real dir
        copy_dir(self.bigmodel_dir, self.real_bigmodel_dir)
        copy_dir(self.result_dir, self.real_result_dir)
        # del slurm logs and tags
        del_file_list_by_patten(self.real_bigmodel_dir, "slurm-*")