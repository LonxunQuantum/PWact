import os
import subprocess
from utils.app_lib.poscar2lammps import p2l
from utils.constant import LAMMPSFILE

def atom_config_to_lammps_in(atom_config_dir:str):
    cwd = os.getcwd()
    os.chdir(atom_config_dir)
    subprocess.run(["config2poscar.x atom.config > /dev/null"], shell = True)
    p2l(output_name = LAMMPSFILE.lammps_sys_config)
    subprocess.run(["rm","atom.config","POSCAR"])
    os.chdir(cwd)

def poscar_to_lammps_in(poscar_dir:str):
    cwd = os.getcwd()
    os.chdir(atom_config_dir)    
    p2l(output_name = LAMMPSFILE.lammps_sys_config)
    subprocess.run(["rm","atom.config","POSCAR"])
    os.chdir(cwd)
    
def traj_to_atom_config(tarj_file:str, atom_save_file:str):
    # read traj_file
    # return atom type name list: such as H, Cu
    raise Exception("ERROR! traj_to_atom_config not realized")

def convert_config_to_mvm(config_list:list[str], mvm_save_file:str):
    # for config in config_list:
    #     content = 
    raise Exception("Error! the method convert_config_to_mvm in app_lib/pwmat.py not realized!")