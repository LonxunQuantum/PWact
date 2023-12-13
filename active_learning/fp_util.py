import os
import numpy as np
import json
import glob

from active_learning.kpu_util import read_kpu_from_csv
from draw_pictures.draw_util import draw_lines

'''
Description: 
param {*} type
    "ab_dir": scf work dir
    "before":get dirs which need to do scf-calculating
    "after": get dirs which have been done scf-calculating
Returns: 
Author: WU Xingxing
'''    
def get_scf_work_list(ab_dir, type="before", sort=True):
    path_list = os.listdir(ab_dir)
    fp_dir_list = []
    for i in path_list:
        atom_config_path = os.path.join(ab_dir, "{}/atom.config".format(i))
        if os.path.exists(atom_config_path):
            if type == "before":
                if os.path.exists(os.path.join(ab_dir, "{}/OUT.ENDIV".format(i))) is False:
                    fp_dir_list.append(os.path.join(ab_dir, "{}".format(i)))
            else:
                if os.path.exists(os.path.join(ab_dir, "{}/OUT.ENDIV".format(i))) is True:
                    fp_dir_list.append(os.path.join(ab_dir, "{}".format(i)))
    if sort:
        fp_dir_list = sorted(fp_dir_list, key=lambda x: int(x.split('/')[-1].split('-')[-1]))
    return fp_dir_list

'''
Description: 
this function like the funtion split_train_dir() in tain_util.py, could be merged
param {*} fp_dirs
param {*} group_size
Returns: 
Author: WU Xingxing
    '''
def split_fp_dirs(fp_dirs, group_size=10):
    start = 0
    res = []
    while start < len(fp_dirs):
        end = start+group_size if start+group_size < len(fp_dirs) else len(fp_dirs)
        res.append(fp_dirs[start:end])
        start += group_size
    return res
    
def get_fp_slurm_scripts(dir):
    #scf_slurm_7.job  scf_success_0.tag
    slurm_job_files = glob.glob(os.path.join(dir, "scf_slurm_*.job"))
    slrum_indexs = [int(job.split('/')[-1].split('.')[0].split('_')[2]) for job in slurm_job_files]

    slurm_job_done_tag = glob.glob(os.path.join(dir, "scf_success_*.tag"))
    tag_indexs = [int(job.split('/')[-1].split('.')[0].split('_')[2]) for job in slurm_job_done_tag]

    res_slurm_job = []
    res_tag = []
    res_done = []
    for i, v in enumerate(slrum_indexs):
        if v in tag_indexs:
            res_done.append(slurm_job_files[i])
        else:
            res_slurm_job.append(slurm_job_files[i])
            res_tag.append(os.path.join(dir, "scf_success_{}.tag".format(v)))
    return res_slurm_job, res_tag, res_done

'''
Description: 
param {*} fp_dir_list
Returns: 
Author: WU Xingxing
'''
def make_scf_slurm_script(fp_dir_list, scf_slurm_path, scf_success_tag, index, gpus=1):
    with open(os.path.join("./template_script_head", "scf.job"), 'r') as rf:
        script_head = rf.readlines()
    cmd = ""
    for i in script_head:
        if "--job-name" in i:
            cmd += "#SBATCH --job-name=scf_{}\n".format(index)
        else:
            cmd += i
    cmd += "\n"
    for i in fp_dir_list:
        cmd += "{\n"
        cmd += "cd {}\n".format(i)
        cmd += "if [ ! -f scf_success.tag ] ; then\n"
        cmd += "    mpirun -np {} PWmat\n".format(gpus)
        cmd += "    if test $? -eq 0; then touch scf_success.tag; else touch error.tag; fi\n"
        cmd += "fi\n"
        cmd += "} &\n\n"
    
    cmd += "wait\n"
    cmd += "\n"
    cmd += "test $? -ne 0 && exit 1\n\n"
    cmd += "echo 0 > {}\n\n".format(scf_success_tag)
    cmd += "\n"
    with open(scf_slurm_path, 'w') as wf:
        wf.write(cmd)
    return scf_slurm_path, scf_success_tag

def read_energy_from_fp_dir(fp_dir):
    fp_list = os.listdir(fp_dir)
    fp_list = sorted(fp_list, key=lambda x: int(x))
    res = {}
    img_idxs = []
    for fp in fp_list:
        energy_file = os.path.join(fp_dir, "{}/OUT.ENDIV".format(fp))
        if os.path.exists(energy_file):
            with open(energy_file, 'r') as rf:
                line = rf.readline()
                energy = float(line.split()[2])
            res[fp] = energy
            img_idxs.append(int(fp.split("-")[-1]))
    return res, img_idxs

def draw_etot():
    save_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000" 
    fp_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/400k_scf"
    res,img_idxs = read_energy_from_fp_dir(fp_dir)
    print(res)
    kpu_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/training/model_dir/md_kpu_dir"
    kpu_info=read_kpu_from_csv(kpu_dir)
    md_etot = kpu_info[['img_idx', 'etot_lab', 'etot_pre']].loc[kpu_info['img_idx'].isin(img_idxs)]
    md_etot['fp_etot'] = list(res.values())[:40]

    draw_lines( [md_etot['img_idx']*4], [abs(md_etot['fp_etot']-md_etot['etot_pre'])], os.path.join(save_dir, "etot_abs_1000.png"),\
        [''], "abs energy error of scf and dpkf, step {} fs".format(1), "abs energy error", "fs")

def read_Ei(dir):
    dirs = os.listdir(dir)
    res = {}
    dir_list = []
    for i in dirs:
        if "image" in i:
            dir_list.append(i)
    dir_list = sorted(dir_list, key=lambda x: int(x.split('_')[-1]))
    for i in dir_list:
        ei = np.load(os.path.join(dir, i, "Ei.npy"))
        res[i] = sum(ei)
    return res

def print_info():
    
    train_e  = read_Ei("/share/home/wuxingxing/datas/al_dir/train_test/md_dft_test/train")
    valid_e  = read_Ei("/share/home/wuxingxing/datas/al_dir/train_test/md_dft_test/valid")

    from utils.separate_movement import MovementOp
    dft = MovementOp("/share/home/wuxingxing/datas/al_dir/train_test/md_dft_test/PWdata/init/MOVEMENT")
    M1_e, M1_force = dft.get_all_images_etot_force("DFT")

    dft = MovementOp("/share/home/wuxingxing/datas/al_dir/train_test/md_dft_test/PWdata/md/MOVEMENT")
    M2_e, M2_force = dft.get_all_images_etot_force("DFT")
    print()

def get_atom_configs():
    iter_res_path = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/iter_result.json"
    iter_result = json.load(open(iter_res_path))
    
def write_x_to_slurm():
    dir = "/share/home/wuxingxing/al_dir/cu_bulk_system/iter.0005/labeling"
    import glob
    slurms = glob.glob(os.path.join(dir, "scf_slurm_*.job"))
    for job in slurms:
        with open(job, 'r') as rf:
            lines = rf.read()
        with open(job, 'w') as wf:
            wf.write(lines)
    
if __name__ =="__main__":
    # draw_etot()
    print_info()