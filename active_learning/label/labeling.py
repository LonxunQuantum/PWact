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

import os, sys, subprocess
import json
import shutil
from math import ceil
from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam, SCFParam
from utils.constant import AL_STRUCTURE, LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, TRAIN_FILE_STRUCTUR, \
    FORCEFILED, ENSEMBLE, LAMMPSFILE, UNCERTAINTY,\
        ELEMENTTABLE
from utils.slurm_script import CONDA_ENV, CPU_SCRIPT_HEAD, GPU_SCRIPT_HEAD, get_slurm_job_run_info, set_slurm_comm_basis, split_job_for_group

from utils.format_input_output import get_iter_from_iter_name, \
    make_md_sys_name, make_temp_press_name, make_temp_name, get_md_template_name, \
        make_train_name, get_traj_file_name,\
            make_scf_name
from utils.file_operation import save_json_file, write_to_file, copy_file, file_read_lines

from utils.app_lib.pwmat import traj_to_atom_config
from active_learning.label.fp_util import get_fp_slurm_scripts, get_scf_work_list, make_scf_slurm_script
from active_learning.slurm import SlurmJob, JobStatus, Mission



from active_learning.workdir import WorkLabDir
from active_learning.util import combine_files
from active_learning.pre_al_data_util import get_movement_num
from active_learning.make_slurm_job_script import make_feature_script_slurm, make_feature_script
from utils.movement2traindata import Scf2Movement
from utils.movement2traindata import movement2traindata

class Labeling(object):
    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.iter = get_iter_from_iter_name(self.itername)
        self.resouce = resource
        self.input_param = input_param
        
        self.md_job = self.input_param.explore.md_job_list[self.iter]
        # md work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        self.md_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.md)
        self.select_dir = os.path.join(self.explore_dir, EXPLORE_FILE_STRUCTURE.select)
        # labed work dir
        self.label_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore) 
        self.scf_dir = os.path.join(self.label_dir, LABEL_FILE_STRUCTURE.scf)
        
    def make_scf_work(self):
        # read select info, and make scf
        scf_dir_list = []
        for md_index, md in enumerate(self.md_job):
            for sys_index, sys in enumerate(md):
                char_len = 3 if len(md.sys_idx) < 1000 else len(str(len(md.sys_idx)))
                md_sys_name = make_md_sys_name(md_index, sys_index, char_len)
                md_sys_dir = os.path.join(self.select_dir, md_sys_name)
                if not os.path.exists(md_sys_dir):
                    continue
                # make scf/md.*.sys.*
                scf_md_sys_dir = os.path.join(self.scf_dir, md_sys_name)
                if not os.path.exists(scf_md_sys_dir):
                    os.makedirs(scf_md_sys_dir)
                for temp_index, temp in enumerate(sys.temp_list):
                    selected_file_list = []
                    if ENSEMBLE.nvt in sys.ensemble:#for nvt ensemble
                        temp_name = make_temp_name(md_index, sys_index, temp_index, char_len)
                        temp_dir = os.path.join(md_sys_dir, temp_name)
                        # file select/md.000.sys.000/md.000.sys.000.t.000.cand
                        temp_candidate_file = "{}/{}.{}".format(md_sys_dir, temp_name, EXPLORE_FILE_STRUCTURE.candidate)
                        if not os.path.exists(temp_candidate_file):
                            continue
                        scf_temp_dir = os.path.join(scf_md_sys_dir, temp_name)
                        if os.path.exists(scf_temp_dir):
                            os.makedirs(scf_temp_dir)
                        if sys.merge_traj:
                            # 合并轨迹之后的处理还没有实现
                            raise Exception("after merge_traij, the method read hold file is not realized!")
                        tarj_dir = os.path.join(temp_dir, EXPLORE_FILE_STRUCTURE.tarj)
                        scf_sub_list = self.make_scf_dir(temp_candidate_file, tarj_dir, scf_temp_dir)
                        scf_dir_list.extend(scf_sub_list)
                        
                    elif ENSEMBLE.npt in sys.ensemble: # for npt ensemble
                        for press_index, press in enumerate(sys.press_list):
                            temp_press_name = make_temp_press_name(md_index, sys_index, temp_index, press_index, char_len)
                            temp_press_dir = os.path.join(md_sys_dir, temp_press_name)
                            # file select/md.000.sys.000/md.000.sys.000.t.000.cand
                            temp_candidate_file = "{}/{}.{}".format(md_sys_dir, temp_press_name, EXPLORE_FILE_STRUCTURE.candidate)
                            if not os.path.exists(temp_candidate_file):
                                continue
                            scf_temp_press_dir = os.path.join(scf_md_sys_dir, temp_name)
                            if os.path.exists(scf_temp_press_dir):
                                os.makedirs(scf_temp_press_dir)
                            if sys.merge_traj:
                                # 合并轨迹之后的处理还没有实现
                                raise Exception("after merge_traij, the method read hold file is not realized!")
                            tarj_dir = os.path.join(temp_dir, EXPLORE_FILE_STRUCTURE.tarj)
                            scf_sub_list = self.make_scf_dir(temp_candidate_file, tarj_dir, scf_temp_dir)
                            scf_dir_list.extend(scf_sub_list)
        return scf_dir_list
    
    def make_scf_dir(self, candidate_file:str, traj_dir:str, scf_dir:str):
        candidate_index = file_read_lines(candidate_file)
        scf_sub_list = []
        for index in candidate_index:
            scf_name = make_scf_name(index)
            scf_sub_dir = os.path.join(scf_dir, scf_name)
            if os.path.exists(scf_sub_dir):
                os.makedirs(scf_sub_dir)
            #1. copy traj file and convert it to atom.config format and get its atom type info
            tarj_file_name = get_traj_file_name(index)
            tarj_file = os.path.join(traj_dir, tarj_file_name)
            atom_config_file = os.path.join(scf_dir, LABEL_FILE_STRUCTURE.atom_config)
            atom_type_list = traj_to_atom_config(tarj_file, atom_config_file)
            #2. copy pseudo potential file
            for atom in atom_type_list:
                pseudo_atom = SCFParam.get_pseudo_by_atom_name(self.input_param.scf.pseudo, atom)
                copy_file(pseudo_atom, os.path.join(scf_sub_dir, os.path.basename(pseudo_atom)))
            #3. make etot.input file
            copy_file(self.input_param.scf.etot_input_file, os.path.join(scf_sub_dir, LABEL_FILE_STRUCTURE.etot_input))
            scf_sub_list.append(scf_sub_dir)
        
        self.make_scf_slurm_job_files(scf_sub_list)
        # return scf_sub_list


    def make_scf_slurm_job_files(self, scf_sub_list:list[str]):
        pwmat_num = self.resouce.pwmat_run_num
        
        groupsize = 1 if self.resouce.scf_resource.group_size is None \
            else self.resouce.scf_resource.group_size
        
        if groupsize > 1:
            groupsize_adj = ceil(groupsize/pwmat_num)
            if groupsize_adj*pwmat_num > groupsize:
                groupsize_adj = ceil(groupsize/pwmat_num)*pwmat_num
                print("the groupsize automatically adjusts from {} to {}".format(groupsize, groupsize_adj))
            else:
                groupsize_adj = groupsize
        sub_list:list[list[str]] = split_job_for_group(groupsize_adj, scf_sub_list)
        
        group_script_path = []
        for group_index, group_jobs in enumerate(sub_list):
            slurm_name = "slurm_{}_{}.job".format(job_index, os.path.basename(job_list[0]))
            slurm_path = os.path.join(os.path.dirname(job_list[0]), slurm_name)
            group_job_script = self.make_scf_slurm_job(group_index, group_jobs, slurm_name)
            write_to_file(slurm_path, group_job_script, "w")
            group_script_path.append(slurm_path)
        return slurm_path

    def make_scf_slurm_job(self, job_index:int, job_list:list[str], job_name:str):
        if job_list[0] == "NONE": # no job in job_list
            return
        
        job_name = slurm_name
        # set head
        script = ""
        if self.resouce.scf_resource.gpu_per_node is None:
            script += CPU_SCRIPT_HEAD.format(job_name, \
                self.resouce.scf_resource.number_node,\
                self.resouce.scf_resource.cpu_per_node,\
                    self.resouce.scf_resource.queue_name)
            mpirun_cmd_template = "mpirun -np {} PWmat".format(self.resouce.scf_resource.cpu_per_node)
            
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, \
                self.resouce.scf_resource.number_node,\
                self.resouce.scf_resource.gpu_per_node,\
                    self.resouce.scf_resource.gpu_per_node,\
                    1,\
                    self.resouce.scf_resource.queue_namee)
            mpirun_cmd_template = "mpirun -np {} PWmat".format(self.resouce.scf_resource.gpu_per_node)
                            
        script += set_slurm_comm_basis(self.resouce.scf_resource.custom_flags, \
            self.resouce.scf_resource.source_list, \
                self.resouce.scf_resource.module_list)
        
        # set conda env
        script += "\n"
        script += CONDA_ENV
        
        script += "\n\n"
        script += "cd {}\n".format(job_dir)
        
        script += "start=$(date +%s)\n"

        scf_cmd = ""
        job_id = 0
        while job_id < len(job_list):
            for i in range(self.self.resouce.pwmat_run_num):
                if job_list[job_id] == "NONE":
                    continue
                scf_cmd += "{\n"
                scf_cmd += "cd {}\n".format(job_list[job_id])
                scf_cmd += "if [ ! -f {} ] ; then\n".format(LABEL_FILE_STRUCTURE.scf_tag)
                scf_cmd += "    {}\n".format(mpirun_cmd_template)
                scf_cmd += "    if test $? -eq 0; then touch {}; else touch {}; fi\n".format(LABEL_FILE_STRUCTURE.scf_tag, LABEL_FILE_STRUCTURE.fail_scf_tag)
                scf_cmd += "fi\n"
                scf_cmd += "} &\n\n"
            scf_cmd += "wait\n\n"
        
        script += scf_cmd
        
        script += "\ntest $? -ne 0 && exit 1\n\n"
        
        script += "end=$(date +%s)\n"
        script += "take=$(( end - start ))\n"
        script += "echo Time taken to execute commands is ${take} seconds > {}\n\n".format(LABEL_FILE_STRUCTURE.scf_tag)
        script += "\n"
        return script

############## old code ########
    # def pre_precess(self):
    #     # read cadidates
    #     kpu_selected = json.load(open(os.path.join(self.system_info["work_root_path"], "kpu_result.json")))[self.itername]
    #     cadidate = kpu_selected["kpu_select"]["cadidate"]
    #     # ln md_dir/md_traj_dir/cadidates
    #     traj_dir = "{}/{}/{}".format(self.work_root_dir, self.itername, "exploring/md_traj_dir")
    #     for i in cadidate:
    #         i = int(i)
    #         i_dir = os.path.join(self.work_dir.ab_dir, "{}-{}".format(self.itername, i))
    #         if os.path.exists(i_dir) is False:
    #             os.makedirs(i_dir)
    #         # ln atom.config
    #         source_atom_config = os.path.abspath(os.path.join(traj_dir, "atom_{}.config".format(i)))
    #         if os.path.exists(os.path.join(i_dir, "atom.config")) is False:
    #             os.symlink(source_atom_config, os.path.join(i_dir, "atom.config"))
            
    #         # ln UPF files
    #         for upf in self.system_info["fp_control"]["UPF"]:
    #             basename = os.path.basename(upf)
    #             if os.path.exists(os.path.join(i_dir, basename)) is False:
    #                 shutil.copy(os.path.abspath(upf), os.path.join(i_dir, basename))#test if need filename
            
    #         # copy etot files
    #         if os.path.exists(os.path.join(i_dir, "etot.input")) is False:
    #             shutil.copy(self.system_info["fp_control"]["etot_input_path"], os.path.join(i_dir, "etot.input"))
    
    # def do_labeling(self):
    #     # prepare files
    #     self.pre_precess()
    #     # run scf job
    #     if "slurm" in self.system_info.keys():
    #         self.do_scf_slurm()
    #     else:
    #         self.do_scf()
    #     self.scf_2_movement()
    #     self.post_precess()

    # def do_scf_slurm(self):
    #     fp_dir_list = get_scf_work_list(self.work_dir.ab_dir, type="before")
    #     if len(fp_dir_list) == 0:
    #         return
    #     # split fp dirs by group_size
    #     group_size = self.system_info["fp_control"]["group_size"]
    #     fp_lists = split_fp_dirs(fp_dir_list, group_size)
    #     mission = Mission()
    #     slurm_jobs, res_tags, res_done = get_fp_slurm_scripts(self.work_dir.work_dir)
    #     fp_done = True if len(slurm_jobs) == 0 and len(res_done) > 0 else False
    #     if fp_done == False:
    #         #recover slurm jobs
    #         if len(slurm_jobs) > 0:
    #             print("recover these SCF Jobs:\n")
    #             print(slurm_jobs)
    #             for i, script_save_path in enumerate(slurm_jobs):
    #                 slurm_cmd = "sbatch {}".format(script_save_path)
    #                 slurm_job = SlurmJob()
    #                 slurm_job.set_tag(res_tags[i])
    #                 slurm_job.set_cmd(slurm_cmd)
    #                 mission.add_job(slurm_job)
    #         # generate new slurm jobs
    #         else:
    #             for i, fp_list in enumerate(fp_lists):
    #                 script_save_path = os.path.join(self.work_dir.work_dir, "scf_slurm_{}.job".format(i))
    #                 tag = os.path.join(self.work_dir.work_dir, "scf_success_{}.tag".format(i))
    #                 script_save_path, tag = make_scf_slurm_script(fp_list, script_save_path, tag, i, self.system_info["fp_control"]["gpus"])
    #                 slurm_cmd = "sbatch {}".format(script_save_path)
    #                 slurm_job = SlurmJob()
    #                 slurm_job.set_tag(tag)
    #                 slurm_job.set_cmd(slurm_cmd)
    #                 mission.add_job(slurm_job)
    #         mission.commit_jobs()
    #         mission.check_running_job()
    #         assert mission.all_job_finished()
    
    # '''
    # Description: do scf work in single node, \
    #     because in mcloud env, the mkl tools cannot loaded on the compute nodes, \
    #     so this work should do by slurm at mgt node. new code at do_labeling_slurrm()
    # param {*} self
    # Returns: 
    # Author: WU Xingxing
    # '''    
    # def do_scf(self):
    #     cwd = os.getcwd()
    #     path_list = os.listdir(self.work_dir.ab_dir)
    #     for i in path_list:
    #         atom_config_path = os.path.join(self.work_dir.ab_dir, "{}/atom.config".format(i))
    #         if os.path.exists(atom_config_path):
    #             if os.path.exists(os.path.join(self.work_dir.ab_dir, "{}/OUT.ENDIV".format(i))) is False:
    #                 os.chdir(os.path.dirname(atom_config_path))
    #                 commands = "mpirun -np {} PWmat".format(self.system_info["fp_control"]["gpus"])
    #                 res = os.system(commands)
    #                 if res != 0:
    #                     raise Exception("run md command {} error!".format(commands))
    #                 os.chdir(cwd)
    #                 # construct the atom.config to MOVEMENT by using REPORT, OUT.FORCE
                
    #     os.chdir(cwd)

    # '''
    # Description: 
    # construct scf output files to movement
    # param {*} self
    # Returns: 
    # Author: WU Xingxing
    # '''
    # def scf_2_movement(self):
    #     fp_dir_list = get_scf_work_list(self.work_dir.ab_dir, type="after")
    #     for i in fp_dir_list:
    #         if os.path.exists(os.path.join(os.path.join(i, "MOVEMENT"))) is False:
    #             atom_config_path = os.path.join(i, "atom.config")
    #             save_movement_path = os.path.join(os.path.join(self.work_dir.ab_dir, "{}/MOVEMENT".format(i)))
    #             if os.path.exists(save_movement_path) is False:
    #                 Scf2Movement(atom_config_path, \
    #                     os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.FORCE".format(i))), \
    #                     os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.ENDIV".format(i))), \
    #                     os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.MLMD".format(i))), \
    #                     save_movement_path)

    # """
    # @Description :
    # set labeling result to iter_result.json:
    # step 1:
    # record new image infos generated in this iter.

    # setp 2:
    # if nums of new images which have not been trained more than the value "data_retrain" in system.config, 
    # they will be converted to the features for trianing:
    # 1. make dir construct: PWdata/MOVEMENT
    # 2. run mlff.py, seper.py, ...
    # 3. record the feature path to the iters which the images belong to
    # @Returns     :
    # @Author       :wuxingxing
    # """
    # def post_precess(self):
    #     iter_result_json_path = "{}/iter_result.json".format(self.work_root_dir)
    #     iter_result_json = json.load(open(iter_result_json_path)) if os.path.exists(iter_result_json_path) else {}

    #     path_list = os.listdir(self.work_dir.ab_dir)

    #     iter_result = {}
    #     iter_result["movement_dir"] = self.work_dir.lab_dpkf_dir

    #     movement_list = []
    #     for i in path_list:
    #         if "iter" not in i:
    #             continue
    #         MOVEMENT_path = os.path.join(self.work_dir.ab_dir, "{}/MOVEMENT".format(i))
    #         if os.path.exists(MOVEMENT_path):
    #             movement_save_path = os.path.join(self.work_dir.lab_dpkf_dir, "{}-{}".format(i, "MOVEMENT"))
    #             if os.path.exists(movement_save_path) is False:
    #                 shutil.copyfile(os.path.abspath(MOVEMENT_path), movement_save_path)
    #             movement_list.append("{}-{}".format(i, "MOVEMENT"))
    #     movement_list = sorted(movement_list, key = lambda x: int(x.split('-')[1]))
    #     iter_result["movement_file"] = movement_list
        
    #     feature_path = os.path.join(self.work_dir.lab_dpkf_dir, "feature_dir")
    #     iter_result["feature_path"] = feature_path
    #     # if new labeled data more than "system_info["data_retrain"]", then make features and retrain at next iter.
    #     if len(movement_list) >= self.system_info["data_retrain"]:
    #         if os.path.exists(feature_path) is False:
    #             os.mkdir(feature_path)
    #         if os.path.exists(os.path.join(feature_path, "PWdata")) is False:
    #             os.mkdir(os.path.join(feature_path, "PWdata"))
    #         # write movements of other iters to one movement file, if target exists, just cover it.
    #         combine_files(self.work_dir.lab_dpkf_dir, movement_list, os.path.join(feature_path, "PWdata/MOVEMENT"))

    #         if os.path.exists(self.work_dir.gen_feat_success_tag) is False:
    #             if "slurm" in self.system_info.keys():
    #                 save_path, tag = make_feature_script_slurm(self.system_info, feature_path, \
    #                                                         self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_slurm_path)
    #                 slurm_cmd = "sbatch {}".format(save_path)
    #                 slurm_job = SlurmJob()
    #                 slurm_job.set_tag(tag)
    #                 slurm_job.set_cmd(slurm_cmd)
    #                 status = slurm_job.running_work()
    #             else:
    #                 make_feature_script(self.system_info, feature_path, \
    #                                                         self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_slurm_path)
    #                 # run cmd
    #                 result = subprocess.call("bash -i {}".format(feature_path), shell=True)
    #                 assert(os.path.exists(self.work_dir.gen_feat_success_tag) == True)
    #         iter_result["retrain"] = True
    #     else:
    #         iter_result["retrain"] = False
    #     iter_result_json[self.itername] = iter_result
    #     json.dump(iter_result_json, open(iter_result_json_path, "w"), indent=4)


        
