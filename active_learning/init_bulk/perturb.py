from matersdk.adalearn.generator.perturbation import BatchPerturbStructure
import os, subprocess
import shutil
from utils.file_operation import copy_file
'''
description: 
param {str} work_dir the dir of atom.config file
param {int} pert_num
param {float} cell_pert_fraction
param {float} atom_pert_distance
return {*}
author: wuxingxing
'''
def do_pertub(atom_config:str, pert_num:int=50, cell_pert_fraction:float=0.03, atom_pert_distance:float=0.01):

    # make pertub dirs
    
    work_dir = os.path.dirname(atom_config)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    atom_config_name = os.path.basename(atom_config)
    cwd = os.getcwd()
    os.chdir(work_dir)
    Perturbed = ['tmp']
    tmp_dir = os.path.join(work_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    copy_file(atom_config, "tmp/atom.config")
    BatchPerturbStructure.batch_perturb(
        Perturbed=Perturbed,
        pert_num=pert_num,
        cell_pert_fraction=cell_pert_fraction,
        atom_pert_distance=atom_pert_distance,
    )
    #Organize the output files, 'mv structures/*.config ..'
    subprocess.run(["mv structures/*.config .. && rm structures -rf"], shell = True)
    os.chdir(cwd)

    # aimd_directory = os.path.join(os.path.abspath(Perturbed[0]), 'AIMD')
    # if not os.path.exists(aimd_directory):
    #     os.makedirs(aimd_directory)

    # # Create 'md-0', 'md-1', ..., 'md49' directories under 'AIMD' directory
    # for i in range(pert_num):
    #     md_directory = os.path.join(aimd_directory, f'md-{i}')
    #     if not os.path.exists(md_directory):
    #         os.makedirs(md_directory)

    #     # Link the corresponding config file from 'structures' directory to 'md-{i}' directory
    #     config_file = os.path.join(os.path.abspath(Perturbed[0]), 'structures', f'{i}.config')
    #     link_file = os.path.join(md_directory, 'atom.config')
    #     if os.path.islink(link_file):
    #         os.remove(link_file)
    #     os.symlink(config_file, link_file)

if __name__ == "__main__":
    do_pertub()