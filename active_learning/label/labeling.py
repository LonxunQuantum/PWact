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
import pandas as pd

from active_learning.user_input.resource import Resource
from active_learning.user_input.iter_input import InputParam, SCFParam
from active_learning.slurm import SlurmJob, Mission

from utils.constant import AL_STRUCTURE, TEMP_STRUCTURE,\
    LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, TRAIN_FILE_STRUCTUR,\
        PWMAT, LAMMPS, SLURM_OUT, INIT_BULK
    
from utils.slurm_script import get_slurm_job_run_info, split_job_for_group, set_slurm_script_content
from utils.format_input_output import get_iter_from_iter_name, get_md_sys_template_name
from utils.file_operation import write_to_file, copy_file, copy_dir, merge_files_to_one, search_files, mv_file, file_shell_op, del_file, del_file_list
from utils.app_lib.pwmat import lammps_dump_to_config, set_etot_input_by_file
from utils.app_lib.common import link_pseudo_by_atom, get_atom_type, link_structure, set_input_script

class Labeling(object):
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
        # labed work dir
        self.label_dir = os.path.join(self.input_param.root_dir, itername, TEMP_STRUCTURE.tmp_run_iter_dir, AL_STRUCTURE.labeling) 
        self.real_label_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.labeling) 
        self.scf_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.scf)
        self.result_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.result)

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
        # read select info, and make scf
        # ["devi_force", "file_path", "config_index"]
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
            tarj_lmp = os.path.join(sub_md_sys_path, EXPLORE_FILE_STRUCTURE.tarj, "{}{}".format(config_index, LAMMPS.traj_postfix))
            traj_target_config = os.path.join(scf_sub_md_sys_path, "{}{}".format(config_index, INIT_BULK.get_postfix(self.resource.dft_style)))
            # lmps traj to atom.config
            sys_index = int(md_sys_name.split(".")[-1])
            sys_atom_config = self.input_param.explore.sys_configs[sys_index]
            atom_type_list, atomic_number_list = get_atom_type(sys_atom_config.sys_config, sys_atom_config.format)
            target_file = lammps_dump_to_config(dump_file=tarj_lmp, save_file = traj_target_config, type_map=atom_type_list, dft_style=self.resource.dft_style)
            self.make_scf_file(scf_sub_md_sys_path, target_file, atom_type_list=atom_type_list)
            scf_dir_list.append(scf_sub_md_sys_path)
            
        self.make_scf_slurm_job_files(scf_dir_list)

    def check_state(self):
        slurm_remain, slurm_done = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_job), \
            tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        return slurm_done

    def do_scf_jobs(self):
        mission = Mission()
        slurm_remain, slurm_done = get_slurm_job_run_info(self.scf_dir, \
            job_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_job), \
            tag_patten="*-{}".format(LABEL_FILE_STRUCTURE.scf_tag))
        slurm_done = True if len(slurm_remain) == 0 and len(slurm_done) > 0 else False
        if slurm_done is False:
            #recover slurm jobs
            if len(slurm_remain) > 0:
                print("Doing these SCF Jobs:\n")
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
                    
    def make_scf_file(self, scf_dir:str, source_config:str, atom_type_list:list):
        # target_config = link_structure(source_config = source_config, 
        #                                 config_format= self.resource.dft_style,
        #                                 target_dir   = scf_dir,
        #                                 dft_style    = self.resource.dft_style)
        # atom_type_list, _ = get_atom_type(target_config, self.resource.dft_style)

        #1. set pseudo files
        link_pseudo_by_atom(self.input_param.scf.pseudo, scf_dir, atom_type_list, self.resource.dft_style)
        #2. make etot.input file
        set_input_script(
            input_file=self.input_param.scf.scf_input_list[0].input_file,
            config=source_config,
            kspacing=self.input_param.scf.scf_input_list[0].kspacing, 
            flag_symm=self.input_param.scf.scf_input_list[0].flag_symm, 
            resource_node=[self.resource.dft_resource.number_node, self.resource.dft_resource.gpu_per_node],
            dft_style=self.resource.dft_style,
            save_dir = scf_dir
        )

    def make_scf_slurm_job_files(self, scf_sub_list:list[str]):
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
                source_list = self.resource.dft_resource.source_list,
                module_list = self.resource.dft_resource.module_list,
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

    '''
    description: 
    collecte OUT.MLMD to mvm-
    param {*} self
    return {*}
    author: wuxingxing
    '''         
    def collect_scf_configs(self):
        aimd_list = []
        md_sys_dir_list = search_files(self.scf_dir, get_md_sys_template_name())
        for md_sys_dir in md_sys_dir_list:
            md_sys_mlmd = []
            sub_md_sys_dir_list =search_files(md_sys_dir, get_md_sys_template_name())
            for sub_md_sys in sub_md_sys_dir_list:
                out_mlmd_list =search_files(sub_md_sys, "*-{}/{}".format(LABEL_FILE_STRUCTURE.scf, INIT_BULK.get_scf_config(self.resource.dft_style)))
                # do a sorted?
                md_sys_mlmd.extend(out_mlmd_list)
            save_file = os.path.join(md_sys_dir, INIT_BULK.get_aimd_config(self.resource.dft_style))
            merge_files_to_one(out_mlmd_list, save_file)
            aimd_list.append(save_file)
        return aimd_list

    def get_aimd_list(self):
        aimd_list = search_files(self.scf_dir, "{}/{}".format(get_md_sys_template_name(), INIT_BULK.get_aimd_config(self.resource.dft_style)))
        return sorted(aimd_list)
    
    '''
    description: 
        需要修改逻辑，不能删除，先复制，然后整体删除
        if reserve_work is True, reserve all temp work dir, copy the result to iter dir
        else:
            if reserve_md_traj is True, copy the trajs
            if reserve_scf_files is True, copy scf files
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def do_post_labeling(self):
        # for training
        # del_file_list(search_files(self.train_dir, "*/{}".format(TRAIN_FILE_STRUCTUR.work_dir)))
        # del_file_list(search_files(self.train_dir, "*/{}".format(TRAIN_FILE_STRUCTUR.train_tag)))
        del_file_list(search_files(self.train_dir, "*/slurm*.out"))
        copy_dir(self.train_dir, self.real_train_dir)

        #for explore dir
        # delete md traj
        md_dirs = search_files(self.explore_dir, "{}/{}/{}".format(EXPLORE_FILE_STRUCTURE.md, get_md_sys_template_name(),get_md_sys_template_name()))
        for md_dir in md_dirs:
            if self.input_param.reserve_md_traj is False:
                del_file_list([os.path.join(md_dir, EXPLORE_FILE_STRUCTURE.tarj)])
            else:
                pass
            del_file(os.path.join(md_dir, LAMMPS.log_lammps))
            del_file(os.path.join(md_dir, EXPLORE_FILE_STRUCTURE.md_tag))
        # delete slurm logs
        md_slurms = search_files(self.explore_dir, "{}/slurm-*".format(EXPLORE_FILE_STRUCTURE.md))
        del_file_list(md_slurms)
        # md_tags = search_files(self.explore_dir, "{}/*-tag*".format(EXPLORE_FILE_STRUCTURE.md))
        # del_file_list(md_tags)
        mv_file(self.explore_dir, self.real_explore_dir)

        # for label dir
        # delete nouse scf files
        if self.input_param.reserve_scf_files is False:
            scf_dirs = search_files(self.scf_dir, "{}/{}/*{}".format(get_md_sys_template_name(),get_md_sys_template_name(), LABEL_FILE_STRUCTURE.scf))
            for scf_dir in scf_dirs:
                scf_files = os.listdir(scf_dir)
                for scf_file in scf_files:
                    if os.path.basename(scf_file) in PWMAT.scf_reserve_list:
                        continue
                    # is *.config but not final.config
                    if PWMAT.config_postfix in os.path.basename(scf_file) and PWMAT.relaxed_config != os.path.basename(scf_file):
                        continue
                    del_file(os.path.join(scf_dir, scf_file))
            # delete tag and logs
            scf_slurms = search_files(self.scf_dir, "slurm-*")
            del_file_list(scf_slurms)
            # scf_tags = search_files(self.scf_dir, "*-tag*")
            # del_file_list(scf_tags)
            mv_file(self.explore_dir, self.real_explore_dir)
        # delete label/result/ format change nouse files
        # 
        # move label dir to main dir
        mv_file(self.label_dir, self.real_label_dir)
        del_file_list([os.path.dirname(self.label_dir)])
