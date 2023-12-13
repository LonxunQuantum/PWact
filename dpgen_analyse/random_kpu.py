import os, glob
import shutil
from active_learning.util import get_random_nums
from active_learning.slurm import SlurmJob, Mission
from active_learning.fp_util import get_scf_work_list, split_fp_dirs, get_fp_slurm_scripts, make_scf_slurm_script

def contruct_scf_work(dir, iter_name, all_nums, need_nums):
    # random list57, 80, 212, 486
    # iter.0004
    cadidate = get_random_nums(0, all_nums, need_nums)
    done_list = get_done_list(dir)
    # set input files:
    rand_dir = os.path.join(dir, "rand_lab")
    if os.path.exists(rand_dir) is False:
        os.makedirs(rand_dir)
    
    explor_dir = os.path.join(os.path.dirname(dir), "exploring", "md_traj_dir")
    target_dir = os.path.join(dir, "rand_lab")
    if os.path.exists(target_dir) is False:
        os.makedirs(target_dir)
    for cad in cadidate:
        # cad done before then copy dir else set scf inputs
        input_dir = os.path.join(target_dir,  "{}-{}".format(iter_name, cad))
        if os.path.exists(input_dir) is True:
            shutil.rmtree(input_dir)
        if cad in done_list:
            #copy data , copy dirs do not need the target dir exist before.
            source_dir = os.path.join(dir, "ab_dir", "{}-{}".format(iter_name, cad))
            shutil.copytree(source_dir, input_dir)
        else:
            os.makedirs(os.path.join(input_dir))
            #set inputs
            # ln atom.config
            source_atom_config = os.path.join(explor_dir, "atom_{}.config".format(cad))
            target_atom_config = os.path.join(input_dir, "atom.config")
            os.symlink(source_atom_config, target_atom_config)
            # copy UPF file
            source_UPF = "/share/home/wuxingxing/datas/system_config/NCPP-SG15-PBE/Cu.SG15.PBE.UPF"
            shutil.copy(source_UPF, input_dir)
            # copy etot file
            source_etot = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/etot.input"
            shutil.copy(source_etot, input_dir)
    return target_dir

def do_scf(work_dir):
    fp_dir_list = get_scf_work_list(work_dir, type="before")
    fp_dir_list = [ "/data"+_[6:] for _ in fp_dir_list]
    if len(fp_dir_list) == 0:
        return
    # split fp dirs by group_size
    group_size = 10
    fp_lists = split_fp_dirs(fp_dir_list, group_size)
    mission = Mission()
    # slurm_jobs, res_tags, res_done = get_fp_slurm_scripts(work_dir)
    slurm_jobs = []
    res_tags = []
    res_done = []
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
                tag = "/data"+tag[6:]
                script_save_path, tag = make_scf_slurm_script(fp_list, script_save_path, tag, i)
                slurm_cmd = "sbatch {}".format(script_save_path)
                slurm_job = SlurmJob()
                slurm_job.set_tag(tag)
                slurm_job.set_cmd(slurm_cmd)
                mission.add_job(slurm_job)
        # mission.commit_jobs()
        # mission.check_running_job()
        # assert mission.all_job_finished()
    print("scf work {} done".format(work_dir))

def get_done_list(dir):
    done_list = glob.glob(os.path.join(dir, "lab_dpkf_dir", "iter.*-MOVEMENT"))
    done = [int(_.split("-")[1]) for _ in done_list]
    done = sorted(done)
    return done

def main():
    #"iter.0002", "iter.0003",  500, 1000,  57, 80, 
    #"iter.0005" , 6000  , 486 cloud doing 
    iter_name = ["iter.0004"]
    all_nums = [3000]
    need_nums = [212]
    for i, v in enumerate(iter_name):
        dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/{}/labeling".format(v)
        target_dir = contruct_scf_work(dir, v, all_nums[i], need_nums[i])
        target_dir = os.path.join(dir, "rand_lab")
        do_scf(target_dir)

if __name__=="__main__":
    main()



