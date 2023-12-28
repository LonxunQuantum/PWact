from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam
from utils.constant import AL_STRUCTURE, EXPLORE_FILE_STRUCTURE

from active_learning.workdir import WorkMDDir
from active_learning.slurm import Mission, SlurmJob, JobStatus
from utils.read_torch_wij_dp import read_torch_dp

from active_learning.pre_al_data_util import get_image_nums
from active_learning.make_slurm_job_script import make_feature_script_slurm, make_feature_script
import os
import json
import sys
import yaml
import shutil
import subprocess
import time
import math

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
class PWmat_MD(object):
    def __init__(self, itername:str, resource: Resource, input_param:InputParam):
        self.itername = itername
        self.resouce = resource
        self.input_param = input_param
        # train work dir
        self.explore_dir = os.path.join(self.input_param.root_dir, itername, AL_STRUCTURE.explore)
        
    def _tmp(self):
        #读取MD控制文件
        self.system_info = json.load(open(sys.argv[1]))
        with open(self.system_info["train_config"]["config_yaml"], "r") as yamlfile:
            self.config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
        work_root_dir = self.system_info["work_root_path"]

        self.iter_result = json.load(open(os.path.join(work_root_dir, "iter_result.json"))) \
        if os.path.exists(os.path.join(work_root_dir, "iter_result.json")) is True \
            else {}

        root_dir = "{}/{}/{}".format(work_root_dir, itername, "exploring")   #cu_bulk_system #cu_4phase_system cuc_system cuo_system
        self.work_dir = WorkMDDir(root_dir)
        
        self.trained_model_path = "{}/{}/{}/{}".format(work_root_dir, itername, "training", "model_dir")
        
        self.curiter = int(self.itername[5:])
        self.md_input_info = self.system_info["md_jobs"][self.curiter]
        self.out_gap = self.md_input_info["out_gap"]
        self.atom_config_path = self.system_info["atom_config"][self.md_input_info["atom_config"]]

    """
    @Description :
    1. ln -s model 
    2. read_torch -> out
    3. mv fread_dfeat/feat.info  vdw_fitB.ntype or ln -s
    4. md.input
    5. run MD: mpirun -n 12 main_MD.x
    @Returns     :
    @Author       :wuxingxing
    """
    def pre_precess(self):
        #convert torch model to pwmat input files
        out_path = self.work_dir.md_dir
        if os.path.exists(os.path.join(out_path, "gen_dp.in")) is False:
            # link model
            source_model_path = os.path.join(self.trained_model_path, "checkpoint.pth.tar")
            if os.path.exists(self.work_dir.model_path) is False:
                os.symlink(os.path.realpath(source_model_path), self.work_dir.model_path)

            # link davg.npy and dstd.npy
            model_pt_path = self.work_dir.model_path
            davg_dstd_dir  = self.system_info["init_data_path"][-1]
            davg_npy_path = os.path.join(davg_dstd_dir, "train/davg.npy")
            dstd_npy_path = os.path.join(davg_dstd_dir, "train/dstd.npy")
            
            #link atom.config
            source_atom_config_path = self.atom_config_path
            if os.path.exists(self.work_dir.atom_config_path) is False:
                os.symlink(os.path.realpath(source_atom_config_path), self.work_dir.atom_config_path)
            atom_config_path = self.work_dir.atom_config_path
            
            read_torch_dp(self.config_yaml, \
                model_pt_path, atom_config_path, out_path, davg_npy_path, dstd_npy_path)

        # write md.input
        if os.path.exists(self.work_dir.md_input_path) is False:
            self.make_md_input()

    def make_md_input(self):
        """ example:
            atom.config
            1, 200, 10, 300, 300 
            F
            5
            1
            2
            31 62
            33 66
        """
        res = "atom.config\n" #header: input file name
        res += "{}, {}, {}, {}, {}\n".format( self.md_input_info["method"], \
                self.md_input_info["MD_steps"], self.md_input_info["step_time"], \
                self.md_input_info["temp_start"], self.md_input_info["temp_end"])
        
        res += "{}\n".format("F")#Place holder
        res += "{}\n".format(5)#Type of model, 5 stands for dp
        res += "{}\n".format(1)#interval for MD movement. No need to change

        atom_info = self.system_info["atom_info"]["atom_type"]
        res += "{}\n".format(len(atom_info))
        for i in atom_info:#[29]
            res += "{} {}\n".format(i, i*2)

        with open(self.work_dir.md_input_path, "w") as wf:
            wf.write(res)
        return True

    def dpkf_md_slurm(self):
        if os.path.exists(self.work_dir.md_success_tag) is True:
            return
        self.pre_precess()
        self.make_md_slurm_script()
        cwd = os.getcwd() #slurm job will change the current dir to md work dir
        # do slurm job
        script_save_path, tag = self.make_md_slurm_script()
        slurm_cmd = "sbatch {}".format(script_save_path)
        slurm_job = SlurmJob()
        slurm_job.set_tag(tag)
        slurm_job.set_cmd(slurm_cmd)
        mission = Mission()
        mission.add_job(slurm_job)
        mission.commit_jobs()
        mission.check_running_job()
        
        os.chdir(cwd)

    def make_md_slurm_script(self):
        md_cpu = int(self.system_info["machine_info"]["md_cpu"]) 
        with open(os.path.join("./template_script_head", "main_MD.job"), 'r') as rf:
            script_head = rf.readlines()
        
        res = ""
        for i in script_head:
            res += i
        res += "\n"
        res += "cd {}\n".format(self.work_dir.md_dir)
        res += "mpirun -n {} main_MD.x".format(md_cpu)
        res += "\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "echo 0 > {}\n\n".format(self.work_dir.md_success_tag)
        
        with open(self.work_dir.md_slurm_path, 'w') as wf:
            wf.write(res)
        return self.work_dir.md_slurm_path, self.work_dir.md_success_tag

    """
    @Description :
        run pwmat with dpkf: mpirun -n 12 main_MD.x
    @Returns     :
    @Author       :wuxingxing
    """
    def dpkf_md(self):
        self.pre_precess()

        cwd = os.getcwd()
        if os.path.exists(os.path.join(self.work_dir.md_dir, "MOVEMENT")):
            return
        os.chdir(self.work_dir.md_dir)
        #run command :mpirun -n 12 main_MD.x 
        md_cpu = int(self.system_info["machine_info"]["md_cpu"]) 
        commands = "mpirun -n {} main_MD.x 1>> run.log 2>>error.log".format(md_cpu)
        res = os.system(commands)
        if res != 0:
            raise Exception("run md command {} error!".format(commands))
        os.chdir(cwd)

    def convert2dpinput(self):
        if os.path.exists(self.work_dir.gen_feat_success_tag):
            return
        # move /md_dpkf/movement to /md_dpkf/Pwdata/MOVEMENT
        source_movement_path = os.path.join(self.work_dir.md_dpkf_dir, "MOVEMENT")
        pwdata_dir = os.path.join(self.work_dir.md_dpkf_dir, "PWdata")
        if os.path.exists(pwdata_dir) is False:
            os.makedirs(pwdata_dir)
        movement_dir = os.path.join(pwdata_dir, "MOVEMENT")
        if os.path.exists(movement_dir) is False:
            os.symlink(os.path.realpath(source_movement_path), movement_dir)
        
        if "slurm" in self.system_info.keys():
            script_save_path, tag = make_feature_script_slurm(self.system_info, self.work_dir.md_dpkf_dir, \
                                                       self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_path)
            slurm_cmd = "sbatch {}".format(script_save_path)
            slurm_job = SlurmJob()
            slurm_job.set_tag(tag)
            slurm_job.set_cmd(slurm_cmd)
            mission = Mission()
            mission.add_job(slurm_job)
            mission.commit_jobs()
            mission.check_running_job()
            
        else:
            make_feature_script(self.system_info, self.work_dir.md_dpkf_dir, \
                                                       self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_path)
            # run cmd
            result = subprocess.call("bash -i {}".format(self.work_dir.gen_feat_path), shell=True)
            assert(os.path.exists(self.work_dir.gen_feat_success_tag) == True)

    '''
    Description: 
        separate train dir to multi dirs, such as: 
            [train0/PWdata, train0/train/image001~image100], [train1/PWdata, train1/train/image101~image200]
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''    
    def separate_train_dir(self):
        #1. get images in train dir
        image_list = get_image_nums(os.path.join(self.work_dir.md_dpkf_dir, "train"))
        split_nums = math.floor(len(image_list) / 100)
        #2. separte image
        index = 0
        split_res = {}
        for split in range(0, split_nums-1):
            split_res[split] = image_list[index: index+100]
        split_res[split_nums-1] = image_list[index:]
        #print(split_res)
        #3. copy images under train dir to new dir
        for split in range(0, split_nums):
            tar_dir = os.path.join(self.work_dir.md_dpkf_dir, "split/train_{}".format(split))
            if os.path.exists(tar_dir) is False:
                os.makedirs(tar_dir)
            image_save_dir = os.path.join(tar_dir, "train")
            if os.path.exists(image_save_dir) is False:
                os.makedirs(image_save_dir)

            for sur in split_res[split]:
                source_dir = os.path.join(self.work_dir.md_dpkf_dir, "train/{}".format(sur))
                shutil(source_dir, image_save_dir)
            
            # link davg.npy and dstd.npy
            davg_dstd_dir  = self.system_info["init_data_path"][-1]
            davg_npy_path = os.path.join(davg_dstd_dir, "train/davg.npy")
            dstd_npy_path = os.path.join(davg_dstd_dir, "train/dstd.npy")
            if os.path.exists(os.path.exists(tar_dir, 'davg.npy')) is False:
                os.symlink(os.path.realpath(davg_npy_path), os.path.exists(image_save_dir, 'davg.npy'))
            if os.path.exists(os.path.exists(tar_dir, 'dstd.npy')) is False:
                os.symlink(os.path.realpath(dstd_npy_path), os.path.exists(image_save_dir, 'dstd.npy'))
            
            
