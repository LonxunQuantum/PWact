# iter4 do:
import os, glob
import shutil
from utils.file_operation import write_to_file, merge_files_to_one, get_random_nums
from active_learning.slurm import SlurmJob, Mission
from active_learning.fp_util import get_scf_work_list, split_fp_dirs, get_fp_slurm_scripts, make_scf_slurm_script
import pandas as pd
import numpy as np

def contruct_scf_work(dir, iter_name, all_nums, need_nums):
    target_dir = os.path.join(dir, "rand_lab")
    if os.path.exists(target_dir) is False:
        os.makedirs(target_dir)
    cadidate = get_random_nums(0, all_nums, need_nums)
    explor_dir = os.path.join(os.path.dirname(dir), "exploring", "md_traj_dir")
    for cad in cadidate:
        # cad done before then copy dir else set scf inputs
        input_dir = os.path.join(target_dir,  "{}-{}".format(iter_name, cad))
        if os.path.exists(input_dir) is True:
            shutil.rmtree(input_dir)
        
        os.makedirs(os.path.join(input_dir))
        #set inputs
        # ln atom.config
        source_atom_config = os.path.join(explor_dir, "atom_{}.config".format(cad))
        target_atom_config = os.path.join(input_dir, "atom.config")
        shutil.copy(source_atom_config, target_atom_config)
        # copy UPF file
        source_UPF = "/share/home/wuxingxing/datas/system_config/NCPP-SG15-PBE/Cu.SG15.PBE.UPF"
        shutil.copy(source_UPF, input_dir)
        # copy etot file
        source_etot = "/share/home/wuxingxing/datas/al_dir/cu_system/etot.input"
        shutil.copy(source_etot, input_dir)
    return target_dir

def do_scf(work_dir):
    fp_dir_list = get_scf_work_list(work_dir, type="before")
    fp_dir_list = ['/data'+_[6:] for _ in fp_dir_list]
    if len(fp_dir_list) == 0:
        return
    # split fp dirs by group_size
    group_size = 2
    fp_lists = split_fp_dirs(fp_dir_list, group_size)
    mission = Mission()
    slurm_jobs, res_tags, res_done = get_fp_slurm_scripts(work_dir)
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
                script_save_path = os.path.join(work_dir, "scf_slurm_{}.job".format(i))
                tag = os.path.join(work_dir, "scf_success_{}.tag".format(i))
                tag = "/data" + tag[6:]
                script_save_path, tag = make_scf_slurm_script(fp_list, script_save_path, tag, i)
                
                slurm_cmd = "sbatch {}".format(script_save_path)
                slurm_job = SlurmJob()
                slurm_job.set_tag(tag)
                slurm_job.set_cmd(slurm_cmd)
                mission.add_job(slurm_job)
        mission.commit_jobs()
        # mission.check_running_job()
        # assert mission.all_job_finished()
    print("scf work {} done".format(work_dir))

def get_done_list(dir):
    done_list = glob.glob(os.path.join(dir, "lab_dpkf_dir", "iter.*-MOVEMENT"))
    done = [int(_.split("-")[1]) for _ in done_list]
    done = sorted(done)
    return done

def main():
    # iter_name = ["iter.0000", "iter.0001", "iter.0002", "iter.0003", "iter.0004", "iter.0005", "iter.0006"] 
    # need_nums = [14,16,349,600,600,287,243]
    # all_nums = [400, 2000, 500, 1000, 3000, 6000, 6000]

    # iter_name = ["iter.0001", "iter.0002", "iter.0003", "iter.0004", "iter.0005", "iter.0006"] 
    # need_nums = [16,349,600,600,287,243]
    # all_nums = [2000, 500, 1000, 3000, 6000, 6000]
    # 6未执行

    # v = "iter.0020" next is 21
    # all = 2000
    # need = 200 
    # all = 2000
    # iter_names = ["iter.0021"]
    # for v in iter_names:
    #     dir = "/share/home/wuxingxing/datas/al_dir/cu_system/{}/labeling".format(v)
    #     target_dir = contruct_scf_work(dir, v, all, need)
    #     do_scf(target_dir)

    need = 200 
    all = 4000
    iter_names = ["iter.0026", "iter.0027", "iter.0028"]
    for v in iter_names:
        dir = "/share/home/wuxingxing/datas/al_dir/cu_system/{}/labeling".format(v)
        target_dir = contruct_scf_work(dir, v, all, need)
        do_scf(target_dir)

def select(kpu_info, low, high):
    accuracy = {}
    cadidate = {}
    error = {}
    for index, row in kpu_info.iterrows():
        img_idx = int(row['img_idx'])
        if row['kpu_res'] <= low:
            accuracy[img_idx] = row['kpu_res']
        elif row["kpu_res"] >= high:
            error[img_idx] = row['kpu_res']
        else:
            cadidate[img_idx] = row['kpu_res']
    kpu_select = {}
    all_nums = kpu_info.shape[0]
    res_info = "accuracy: {} {}    cadidate: {} {}   error: {} {}".format(
        len(accuracy.keys()), round(len(accuracy.keys()) / all_nums, 2), 
        len(cadidate.keys()), round(len(cadidate.keys()) / all_nums, 2),
        len(error.keys()), round(len(error.keys()) / all_nums, 2)
    )
    kpu_select['res_info'] = res_info
    kpu_select['accuracy'] = accuracy
    kpu_select['cadidate'] = cadidate
    kpu_select['error'] = error

    return res_info, kpu_select

def tmp_convert_scf2movement():
    from utils.movement2traindata import Scf2Movement
    from dpgen_analyse.make_slurm_job_script import make_feature_script_slurm

    ab_dirs = ["/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0019/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0020/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0021/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0022/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0023/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0024/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0026/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0027/labeling/rand_lab",
                "/share/home/wuxingxing/datas/al_dir/cu_system/rand_test_i28/iter.0028/labeling/rand_lab"]
    for ab_dir in ab_dirs:
        feat_dir = os.path.join(ab_dir,"feat_dir")
        if os.path.exists(feat_dir) is True:
            shutil.rmtree(feat_dir)
        if os.path.exists(feat_dir) is False:
            os.makedirs(feat_dir)

        fp_dir_list = get_scf_work_list(ab_dir, type="after")
        print("{} has {} images".format(ab_dir, len(fp_dir_list)))
        for i in fp_dir_list:
            if os.path.exists(os.path.join(os.path.join(i, "MOVEMENT"))) is False:
                atom_config_path = os.path.join(i, "atom.config")
                save_movement_path = os.path.join(os.path.join(ab_dir, "{}/MOVEMENT".format(i)))
                if os.path.exists(save_movement_path) is False:
                    Scf2Movement(atom_config_path, \
                        os.path.join(os.path.join(ab_dir, "{}/OUT.FORCE".format(i))), \
                        os.path.join(os.path.join(ab_dir, "{}/OUT.ENDIV".format(i))), \
                        os.path.join(os.path.join(ab_dir, "{}/OUT.MLMD".format(i))), \
                        save_movement_path)
                    
        path_list = os.listdir(ab_dir)

        movement_list = []
        for i in path_list:
            if "iter" not in i:
                continue
            MOVEMENT_path = os.path.join(ab_dir, "{}/MOVEMENT".format(i))
            if os.path.exists(MOVEMENT_path):
                movement_save_path = os.path.join(feat_dir, "{}-{}".format(i, "MOVEMENT"))
                if os.path.exists(movement_save_path) is False:
                    shutil.copyfile(os.path.abspath(MOVEMENT_path), movement_save_path)
                movement_list.append("{}-{}".format(i, "MOVEMENT"))
        movement_list = sorted(movement_list, key = lambda x: int(x.split('-')[1]))
        
        # if new labeled data more than "system_info["data_retrain"]", then make features and retrain at next iter.
        if os.path.exists(os.path.join(feat_dir, "PWdata")) is False:
            os.mkdir(os.path.join(feat_dir, "PWdata"))
        # write movements of other iters to one movement file, if target exists, just cover it.
        merge_files_to_one(feat_dir, movement_list, os.path.join(feat_dir, "PWdata/MOVEMENT"))

def contruct_movements():
    ab_dirs = glob.glob(os.path.join("/data/home/wuxingxing/al_dir/si_2", "iter.0*"))
    ab_dirs = sorted(ab_dirs, key= lambda x: int(x.split('/')[-1].split('.')[-1]))
    save_dir = os.path.join("/share/home/wuxingxing/al_dir/si_2", "final")
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    for ab_dir in ab_dirs:
        movement_list = glob.glob(os.path.join(ab_dir, "labeling/lab_dpkf_dir/*-MOVEMENT"))
        if  len(movement_list) == 0:
            continue
        movement_list = sorted(movement_list, key= lambda x: int(x.split('/')[-1].split('-')[1]))

        itername = int(os.path.basename(ab_dir).split('.')[-1])
        save_path = os.path.join(save_dir, "{}_{}_MOVEMENT".format(itername, len(movement_list)))
        merge_files_to_one(None, movement_list, save_path)
        print("{} saved".format(os.path.basename(save_path)))

def test_iter_done():
    import os, glob
    dir = ""
    iters = glob.glob(os.path.join(dir, "iter.0006*"))
    res = []
    for iter in iters:
        ENDIV = os.path.join(iter, "OUT.ENDIV")
        print(ENDIV)
        if os.path.exists(ENDIV) is False:
            res.append(iter)
    print("these work done error:")
    print(res)

'''
Description: 
check dpgen jobs which run errors.
Returns: 
Author: WU Xingxing
'''
def check_jobs():
    import argparse, glob, os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='specify input scf dir', type=str, default='.')
    args = parser.parse_args()
    dir = args.dir
    task_list = glob.glob(os.path.join(dir, "task.*"))
    job_list = glob.glob(os.path.join(dir, "*.sub"))
    res = {}
    all_task = []
    for job in job_list:
        job_id = os.path.basename(job)
        tasks = []
        with open(job, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            if 'cd task' in line:
                tasks.append(line.split(' ')[1].strip())
        all_task.extend(tasks)
        error = []
        for task in tasks:
            tag = glob.glob(os.path.join(dir, task, '*task_tag_finished'))
            if len(tag) == 0:
                error.append(task.split('.')[0])
        if len(error) > 0:
            res[job_id].append(error)
    print(res)

def check_pwmat_jobs():
    import argparse, glob, os
    from subprocess import Popen, PIPE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='specify input scf dir', type=str, default='.')
    args = parser.parse_args()
    dir = args.dir
    job_list = glob.glob(os.path.join(dir, "*.job"))
    res = []
    all_task = []
    for job in job_list:
        job_id = os.path.basename(job)
        tasks = []
        with open(job, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            if 'cd ' in line:
                tasks.append(line.split(' ')[-1].split('/')[-1].strip())
        all_task.extend(tasks)
        error = []
        for task in tasks:
            if os.path.exists(os.path.join(dir, task, 'OUT.ENDIV')) is False:
                error.append(task)
                cmd1 = "rm {}/{}/scf_success.tag".format(dir, task)
                os.system(cmd1)

        if len(error) > 0:
            res.append(job_id)
    print("all tasks: ", len(all_task))
    res = sorted(res, key=lambda x: int(x.split('.')[0].split('_')[2]))
    print("work error:\n", res)
    os.chdir(dir)
    for job in res:
        cmd = "sbatch {}".format(job)
        ret = Popen([cmd], stdout=PIPE, stderr=PIPE, shell = True)
        stdout, stderr = ret.communicate()
        if str(stderr, encoding='ascii') != "":
            raise RuntimeError (stderr)
        job_id = str(stdout, encoding='ascii').replace('\n','').split()[-1]
        print ("job {} submitted, job id {}".format(job, job_id))

if __name__ == "__main__":
    # tmp_convert_scf2movement()
    # check_pwmat_jobs()
    # main()
    contruct_movements()