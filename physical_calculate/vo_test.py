import argparse
import os, sys, glob
import json
import shutil
import math

from active_learning.util import write_to_file, combine_files
from active_learning.fp_util import get_scf_work_list, split_fp_dirs, get_fp_slurm_scripts, make_scf_slurm_script
from utils.movement2traindata import Scf2Movement

def change_poscar_lattice(baselines, lat):
    for i in [2, 3, 4]:
        line = baselines[i]
        tmp = line.split()
        if i == 2:
            newline = "        " + "{}".format(lat) + "         " + tmp[1]+ "         " + tmp[2] + "\n"
        elif i == 3:
            newline = "        " + tmp[0] + "         " + "{}".format(lat) + "         " + tmp[2] + "\n"
        else:
            newline = "        " + tmp[0] + "         " + tmp[1] + "         " + "{}".format(lat) + "\n"
        baselines[i] = newline
    return baselines

def make_v0_posars(config):
    cell_poscar_path = config["cell_poscar_path"]
    work_dir = config["work_dir"]
    lattice = config["lattice"]
    nums = config["nums"]
    gap = config["gap"]

    os.chdir(work_dir)
    # lattice_max = round(lattice*(1+gap)+gap,2)
    # lattice_min = round(lattice*(1-gap)-gap,2)
    lattice_max = 3.8
    lattice_min = 3.61
    lattices = []
    while lattice_min <= lattice_max:
        lattices.append(round(lattice_min+gap, 2))
        lattice_min = round(lattice_min+gap, 2)
    for lat in lattices:
        with open(cell_poscar_path, 'r') as rf:
            baselines = rf.readlines()
        newlines = change_poscar_lattice(baselines, lat)
        save_dir = os.path.join(work_dir, '{}'.format(lat))
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "POSCAR"), 'w') as wf:
            wf.writelines(newlines)
        # convert poscar to atom.config
        cwd = os.getcwd()
        os.chdir(save_dir)
        os.system("poscar2config.x POSCAR")
        os.chdir(cwd)
        # copy etot.input, pbe file and scf job
        shutil.copy(config["etot_input_path"], os.path.join(save_dir, "etot.input"))
        for UPF in config["UPF"]:
            shutil.copy(UPF, os.path.join(save_dir, os.path.basename(UPF)))
        # copy scf job
        # shutil.copy(config["slurm_job"], os.path.join(save_dir, "scf.job"))
        set_scf_job(save_dir)

def set_scf_job(work_dir):
    res = ""
    res += "#!/bin/sh\n"
    res += "#SBATCH --job-name=scf_0\n"
    res += "#SBATCH --nodes=1\n"
    res += "#SBATCH --ntasks-per-node=1\n"
    res += "#SBATCH --gres=gpu:1\n"
    res += "#SBATCH --gpus-per-task=1\n"
    res += "#SBATCH --partition=a100,3090\n"
    res += "module load compiler/2022.0.2\n"
    res += "module load mkl/2022.0.2\n"
    res += "module load mpi/2021.5.1\n"
    res += "module load cuda/11.6\n"
    res += "cd {}\n".format(work_dir)
    res += "mpirun -np {} PWmat\n".format(1)
    res += "if test $? -eq 0; then touch scf_success.tag; else touch error.tag; fi\n"
    scf_slurm_path = "{}/{}".format(work_dir, "scf.job")
    with open(scf_slurm_path, 'w') as wf:
        wf.write(res)
    tag = "{}/{}".format(work_dir, "scf_success.tag")
    return scf_slurm_path, tag

def tmp_convert_scf2movement(config):
    work_dir = config["work_dir"]
    os.chdir(work_dir)
    fp_dir_list = get_scf_work_list(work_dir, type="after", sort=False)
    print("{} has {} images".format(work_dir, len(fp_dir_list)))
    movement_list = []
    for i in fp_dir_list:
        atom_config_path = os.path.join(i, "atom.config")
        save_movement_path = os.path.join(i, "MOVEMENT")
        if os.path.exists(save_movement_path) is False:
            Scf2Movement(atom_config_path, \
                os.path.join(i, "OUT.FORCE"), \
                os.path.join(i, "OUT.ENDIV"), \
                os.path.join(i, "OUT.MLMD"), \
                save_movement_path)
        movement_list.append(save_movement_path)
    movement_list = sorted(movement_list, key = lambda x: float(x.split('/')[-2]))
    if len(movement_list) < 20:
        movement_list.extend([movement_list[-1] for _ in range(0, 20-len(movement_list),1)])
    # write movements of other iters to one movement file, if target exists, just cover it.
    combine_files(None, movement_list, os.path.join(work_dir, "MOVEMENT"))

def get_ep(config, type="dft"):
    work_dir = config["work_dir"]
    mlmds = glob.glob(os.path.join(work_dir, "*", "OUT.MLMD"))
    mlmds = sorted(mlmds, key=lambda x: float(x.split('/')[-2]))
    Ep_list = []
    for mlmd in mlmds:
        print('ep dir is {}'.format(os.path.dirname(mlmd)))
        with open(mlmd, 'r') as rf:
            lines = rf.readlines()
        Ep = float(lines[0].split("=")[2].strip().split()[1])
        Ep_list.append(Ep)
    return Ep_list

def get_volume(config, type="dft"):
    work_dir = config["work_dir"]
    atom_cofigs = glob.glob(os.path.join(work_dir, "*", "atom.config"))
    atom_cofigs = sorted(atom_cofigs, key=lambda x: float(x.split('/')[-2]))
    volumes = []
    for at in atom_cofigs:
        print('volume dir is {}'.format(os.path.dirname(at)))
        with open(at, 'r') as rf:
            lines = rf.readlines()
        x = float(lines[2].split()[0].strip())
        y = float(lines[3].split()[1].strip())
        z = float(lines[4].split()[2].strip())
        volumes.append(x*y*z)
    return volumes

def get_ep_volume(config):
    eps = get_ep(config, type="dft")
    vols = get_volume(config, type="dft")
    work_dir = config["work_dir"]
    with open(os.path.join(work_dir, "summary"), 'w') as wf:
        for i in range(0, len(eps)):
            wf.write("{} {}\n".format(vols[i], eps[i]))

if __name__=="__main__":
    config = json.load(open(sys.argv[1]))
    type = config["type"]
    make_v0_posars(config)
    # get_ep_volume(config)
    # tmp_convert_scf2movement(config)    #atom.configs to movement ->feature for kpu or dpgen
