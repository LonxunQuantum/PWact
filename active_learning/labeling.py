import os, sys, subprocess
import json
import shutil
from active_learning.workdir import WorkLabDir
from active_learning.util import write_to_file, combine_files
from active_learning.fp_util import get_fp_slurm_scripts, get_scf_work_list, split_fp_dirs, make_scf_slurm_script
from active_learning.pre_al_data_util import get_movement_num
from active_learning.slurm import SlurmJob, JobStatus, Mission
from active_learning.make_slurm_job_script import make_feature_script_slurm, make_feature_script
from utils.movement2traindata import Scf2Movement
from utils.movement2traindata import movement2traindata

class Labeling(object):
    def __init__(self, itername):
        self.itername = itername

        self.system_info = json.load(open(sys.argv[1]))
        self.work_root_dir = self.system_info["work_root_path"]
        
        self.work_dir = WorkLabDir("{}/{}/{}".format(self.work_root_dir, itername, "labeling"))
        
        self.curiter = int(self.itername[5:])
        self.md_input_info = self.system_info["iter_control"][self.curiter]
        self.atom_config_path = self.system_info["atom_config"][self.md_input_info["atom_config"]]

    def pre_precess(self):
        # read cadidates
        kpu_selected = json.load(open(os.path.join(self.system_info["work_root_path"], "kpu_result.json")))[self.itername]
        cadidate = kpu_selected["kpu_select"]["cadidate"]
        # ln md_dir/md_traj_dir/cadidates
        traj_dir = "{}/{}/{}".format(self.work_root_dir, self.itername, "exploring/md_traj_dir")
        for i in cadidate:
            i = int(i)
            i_dir = os.path.join(self.work_dir.ab_dir, "{}-{}".format(self.itername, i))
            if os.path.exists(i_dir) is False:
                os.makedirs(i_dir)
            # ln atom.config
            source_atom_config = os.path.abspath(os.path.join(traj_dir, "atom_{}.config".format(i)))
            if os.path.exists(os.path.join(i_dir, "atom.config")) is False:
                os.symlink(source_atom_config, os.path.join(i_dir, "atom.config"))
            
            # ln UPF files
            for upf in self.system_info["fp_control"]["UPF"]:
                basename = os.path.basename(upf)
                if os.path.exists(os.path.join(i_dir, basename)) is False:
                    shutil.copy(os.path.abspath(upf), os.path.join(i_dir, basename))#test if need filename
            
            # copy etot files
            if os.path.exists(os.path.join(i_dir, "etot.input")) is False:
                shutil.copy(self.system_info["fp_control"]["etot_input_path"], os.path.join(i_dir, "etot.input"))
    
    def do_labeling(self):
        # prepare files
        self.pre_precess()
        # run scf job
        if "slurm" in self.system_info.keys():
            self.do_scf_slurm()
        else:
            self.do_scf()
        self.scf_2_movement()
        self.post_precess()

    def do_scf_slurm(self):
        fp_dir_list = get_scf_work_list(self.work_dir.ab_dir, type="before")
        if len(fp_dir_list) == 0:
            return
        # split fp dirs by group_size
        group_size = self.system_info["fp_control"]["group_size"]
        fp_lists = split_fp_dirs(fp_dir_list, group_size)
        mission = Mission()
        slurm_jobs, res_tags, res_done = get_fp_slurm_scripts(self.work_dir.work_dir)
        fp_done = True if len(slurm_jobs) == 0 and len(res_done) > 0 else False
        if fp_done == False:
            #recover slurm jobs
            if len(slurm_jobs) > 0:
                print("recover these SCF Jobs:\n")
                print(slurm_jobs)
                for i, script_save_path in enumerate(slurm_jobs):
                    slurm_cmd = "sbatch {}".format(script_save_path)
                    slurm_job = SlurmJob()
                    slurm_job.set_tag(res_tags[i])
                    slurm_job.set_cmd(slurm_cmd)
                    mission.add_job(slurm_job)
            # generate new slurm jobs
            else:
                for i, fp_list in enumerate(fp_lists):
                    script_save_path = os.path.join(self.work_dir.work_dir, "scf_slurm_{}.job".format(i))
                    tag = os.path.join(self.work_dir.work_dir, "scf_success_{}.tag".format(i))
                    script_save_path, tag = make_scf_slurm_script(fp_list, script_save_path, tag, i, self.system_info["fp_control"]["gpus"])
                    slurm_cmd = "sbatch {}".format(script_save_path)
                    slurm_job = SlurmJob()
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd)
                    mission.add_job(slurm_job)
            mission.commit_jobs()
            mission.check_running_job()
            assert mission.all_job_finished()
    
    '''
    Description: do scf work in single node, \
        because in mcloud env, the mkl tools cannot loaded on the compute nodes, \
        so this work should do by slurm at mgt node. new code at do_labeling_slurrm()
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''    
    def do_scf(self):
        cwd = os.getcwd()
        path_list = os.listdir(self.work_dir.ab_dir)
        for i in path_list:
            atom_config_path = os.path.join(self.work_dir.ab_dir, "{}/atom.config".format(i))
            if os.path.exists(atom_config_path):
                if os.path.exists(os.path.join(self.work_dir.ab_dir, "{}/OUT.ENDIV".format(i))) is False:
                    os.chdir(os.path.dirname(atom_config_path))
                    commands = "mpirun -np {} PWmat".format(self.system_info["fp_control"]["gpus"])
                    res = os.system(commands)
                    if res != 0:
                        raise Exception("run md command {} error!".format(commands))
                    os.chdir(cwd)
                    # construct the atom.config to MOVEMENT by using REPORT, OUT.FORCE
                
        os.chdir(cwd)

    '''
    Description: 
    construct scf output files to movement
    param {*} self
    Returns: 
    Author: WU Xingxing
    '''
    def scf_2_movement(self):
        fp_dir_list = get_scf_work_list(self.work_dir.ab_dir, type="after")
        for i in fp_dir_list:
            if os.path.exists(os.path.join(os.path.join(i, "MOVEMENT"))) is False:
                atom_config_path = os.path.join(i, "atom.config")
                save_movement_path = os.path.join(os.path.join(self.work_dir.ab_dir, "{}/MOVEMENT".format(i)))
                if os.path.exists(save_movement_path) is False:
                    Scf2Movement(atom_config_path, \
                        os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.FORCE".format(i))), \
                        os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.ENDIV".format(i))), \
                        os.path.join(os.path.join(self.work_dir.ab_dir, "{}/OUT.MLMD".format(i))), \
                        save_movement_path)

    """
    @Description :
    set labeling result to iter_result.json:
    step 1:
    record new image infos generated in this iter.

    setp 2:
    if nums of new images which have not been trained more than the value "data_retrain" in system.config, 
    they will be converted to the features for trianing:
    1. make dir construct: PWdata/MOVEMENT
    2. run mlff.py, seper.py, ...
    3. record the feature path to the iters which the images belong to
    @Returns     :
    @Author       :wuxingxing
    """
    def post_precess(self):
        iter_result_json_path = "{}/iter_result.json".format(self.work_root_dir)
        iter_result_json = json.load(open(iter_result_json_path)) if os.path.exists(iter_result_json_path) else {}

        path_list = os.listdir(self.work_dir.ab_dir)

        iter_result = {}
        iter_result["movement_dir"] = self.work_dir.lab_dpkf_dir

        movement_list = []
        for i in path_list:
            if "iter" not in i:
                continue
            MOVEMENT_path = os.path.join(self.work_dir.ab_dir, "{}/MOVEMENT".format(i))
            if os.path.exists(MOVEMENT_path):
                movement_save_path = os.path.join(self.work_dir.lab_dpkf_dir, "{}-{}".format(i, "MOVEMENT"))
                if os.path.exists(movement_save_path) is False:
                    shutil.copyfile(os.path.abspath(MOVEMENT_path), movement_save_path)
                movement_list.append("{}-{}".format(i, "MOVEMENT"))
        movement_list = sorted(movement_list, key = lambda x: int(x.split('-')[1]))
        iter_result["movement_file"] = movement_list
        
        feature_path = os.path.join(self.work_dir.lab_dpkf_dir, "feature_dir")
        iter_result["feature_path"] = feature_path
        # if new labeled data more than "system_info["data_retrain"]", then make features and retrain at next iter.
        if len(movement_list) >= self.system_info["data_retrain"]:
            if os.path.exists(feature_path) is False:
                os.mkdir(feature_path)
            if os.path.exists(os.path.join(feature_path, "PWdata")) is False:
                os.mkdir(os.path.join(feature_path, "PWdata"))
            # write movements of other iters to one movement file, if target exists, just cover it.
            combine_files(self.work_dir.lab_dpkf_dir, movement_list, os.path.join(feature_path, "PWdata/MOVEMENT"))

            if os.path.exists(self.work_dir.gen_feat_success_tag) is False:
                if "slurm" in self.system_info.keys():
                    save_path, tag = make_feature_script_slurm(self.system_info, feature_path, \
                                                            self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_slurm_path)
                    slurm_cmd = "sbatch {}".format(save_path)
                    slurm_job = SlurmJob()
                    slurm_job.set_tag(tag)
                    slurm_job.set_cmd(slurm_cmd)
                    status = slurm_job.running_work()
                else:
                    make_feature_script(self.system_info, feature_path, \
                                                            self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_slurm_path)
                    # run cmd
                    result = subprocess.call("bash -i {}".format(feature_path), shell=True)
                    assert(os.path.exists(self.work_dir.gen_feat_success_tag) == True)
            iter_result["retrain"] = True
        else:
            iter_result["retrain"] = False
        iter_result_json[self.itername] = iter_result
        json.dump(iter_result_json, open(iter_result_json_path, "w"), indent=4)


        
