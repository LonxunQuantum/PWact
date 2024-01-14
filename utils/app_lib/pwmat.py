import os
import numpy as np
import subprocess
from utils.app_lib.poscar2lammps import p2l
from utils.constant import LAMMPSFILE, ELEMENTTABLE_2, PWMAT

def poscar_to_atom_config(poscar_dir:str, input_name:str="POSCAR", save_name:str="poscar_to_atom.config"):
    cwd = os.getcwd()
    os.chdir(poscar_dir)
    subprocess.run(["poscar2config.x {} > /dev/null".format(input_name)], shell = True)
    subprocess.run(["mv {} {} > /dev/null".format(PWMAT.atom_config, save_name)], shell = True)
    os.chdir(cwd)

def atom_config_to_poscar(atom_config_dir:str, input_name:str="atom.config", save_name:str="atom_to_POSCAR"):
    cwd = os.getcwd()
    os.chdir(atom_config_dir)
    subprocess.run(["config2poscar.x {} > /dev/null".format(input_name)], shell = True)
    # remove the black space in ' Selective Dynamics' line in POSCAR file
    subprocess.run(["sed -i '/^ *Selective Dynamics/s/^ *//' {} > /dev/null".format(LAMMPSFILE.poscar)], shell = True)
    # remame the poscar file
    subprocess.run(["mv {} {} > /dev/null".format(LAMMPSFILE.poscar, save_name)], shell = True)
    os.chdir(cwd)
     
def atom_config_to_lammps_in(atom_config_dir:str, atom_config_name:str="atom.config"):
    cwd = os.getcwd()
    os.chdir(atom_config_dir)
    subprocess.run(["config2poscar.x {} > /dev/null".format(atom_config_name)], shell = True)
    p2l(output_name = LAMMPSFILE.lammps_sys_config)
    subprocess.run(["rm","atom.config","POSCAR"])
    os.chdir(cwd)
    
def poscar_to_lammps_in(poscar_dir:str):
    cwd = os.getcwd()
    os.chdir(poscar_dir)
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


def _reciprocal_box(box):
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    return rbox


def _make_pwmat_kp_mp(kpoints):
    ret = ""
    ret += "%d %d %d 0 0 0 " % (kpoints[0], kpoints[1], kpoints[2])
    return ret


def _make_kspacing_kpoints(config, kspacing):
    with open(config, "r") as fp:
        lines = fp.read().split("\n")
    box = []
    for idx, ii in enumerate(lines):
        if "LATTICE" in ii.upper():
            for kk in range(idx + 1, idx + 1 + 3):
                vector = [float(jj) for jj in lines[kk].split()[0:3]]
                box.append(vector)
            box = np.array(box)
            rbox = _reciprocal_box(box)
    kpoints = [
        (np.ceil(2 * np.pi * np.linalg.norm(ii) / kspacing).astype(int)) for ii in rbox
    ]
    ret = _make_pwmat_kp_mp(kpoints)
    return ret


def make_pwmat_input_dict(
    node1,
    node2,
    job_type,
    atom_config,
    pseudo_list,
    ecut,
    ecut2,
    e_error,
    rho_error,
    out_force,
    energy_decomp,
    out_stress,
    icmix=None,
    smearing=None,
    sigma=None,
    kspacing=0.5,
    flag_symm=None,
    out_wg="F",
    out_rho="F",
    out_mlmd="T",
    relax_detail=None,
    vdw = None
):
    icmix, smearing, sigma = _make_smearing(icmix, smearing, sigma)
    flag_symm = _make_flag_symm(flag_symm)
        
    script = ""
    script += "{} {}\n".format(node1, node2)
    script += "job = {}\n".format(job_type)
    script += "{} = {}\n".format("in.atom", os.path.basename(atom_config))
    for pse_index, pseudo in enumerate(pseudo_list):
        script += "IN.PSP{} = {}\n".format(pse_index+1, pseudo)
    script += "{} =  {}\n".format("ecut", ecut)
    script += "{} = {}\n".format("ecut2", ecut2)
    script += "{} = {}\n".format("e_error", e_error)
    script += "{} = {}\n".format("rho_error", rho_error)
    script += "{} = {}\n".format("out.force", out_force)
    script += "{} = {}\n".format("energy_decomp", energy_decomp)
    script += "{} = {}\n".format("out.stress", out_stress)
    if icmix is not None:
        if sigma is not None:
            if smearing is not None:
                SCF_ITER0_1 = "6 4 3 0.0000 " + str(sigma) + " " + str(smearing)
                SCF_ITER0_2 = (
                    "94 4 3 " + str(icmix) + " " + str(sigma) + " " + str(smearing)
                )
            else:
                SCF_ITER0_1 = "6 4 3 0.0000 " + str(sigma) + " 2"
                SCF_ITER0_2 = "94 4 3 " + str(icmix) + " " + str(sigma) + " 2"

        else:
            if smearing is not None:
                SCF_ITER0_1 = "6 4 3 0.0000 0.025 " + str(smearing)
                SCF_ITER0_2 = "94 4 3 " + str(icmix) + " 0.025 " + str(smearing)
            else:
                SCF_ITER0_1 = "6 4 3 0.0000 0.025 2"
                SCF_ITER0_2 = "94 4 3 " + str(icmix) + " 0.025 2"
    else:
        if sigma is not None:
            if smearing is not None:
                SCF_ITER0_1 = "6 4 3 0.0000 " + str(sigma) + " " + str(smearing)
                SCF_ITER0_2 = "94 4 3 1.0000 " + str(sigma) + " " + str(smearing)
            else:
                SCF_ITER0_1 = "6 4 3 0.0000 " + str(sigma) + " 2"
                SCF_ITER0_2 = "94 4 3 1.0000 " + str(sigma) + " 2"
        else:
            if smearing is not None:
                SCF_ITER0_1 = "6 4 3 0.0000 0.025 " + str(smearing)
                SCF_ITER0_2 = "94 4 3 1.0000 0.025 " + str(smearing)
            else:#run this config
                SCF_ITER0_1 = "6 4 3 0.0000 0.025 2"
                SCF_ITER0_2 = "94 4 3 1.0000 0.025 2"
    script += "{} = {}\n".format("scf_iter0_1", SCF_ITER0_1)
    script += "{} = {}\n".format("scf_iter0_2", SCF_ITER0_2)
    if flag_symm is not None:
        MP_N123 = _make_kspacing_kpoints(atom_config, kspacing)
        MP_N123 += str(flag_symm)
    else:
        MP_N123 = _make_kspacing_kpoints(atom_config, kspacing)
    # use pwmat defualt value
    script += "{} = {}\n".format("mp_n123", MP_N123)
    script += "{} = {}\n".format("out.wg", out_wg)
    script += "{} = {}\n".format("out.rho", out_rho)
    script += "{} = {}\n".format("out.mlmd", out_mlmd)
    # if do relax
    if job_type == PWMAT.relax:
        if relax_detail is not None:
            script += "{} = {}\n".format("relax_detail", relax_detail)
        if vdw is not None:
            script += "{} = {}\n".format("vdw", vdw)
    return script

def _make_smearing(icmix=None, smearing = None, sigma = None):
    if icmix is None:
        if smearing is None:
            if sigma is None:
                return None, None, None
            else:
                return None, None, sigma
        else:
            if sigma is None:
                return None, smearing, None
            else:
                return None, smearing, sigma
    else:
        if smearing is None:
            if sigma is None:
                return icmix, None, None
            else:
                return icmix, None, sigma
        else:
            if sigma is None:
                return icmix, smearing, None
            else:
                return icmix, smearing, sigma
    

def _make_flag_symm(flag_symm = None):
    if flag_symm is None:
        return None
    if flag_symm == "NONE":
        flag_symm = None
    elif str(flag_symm) not in [None, "0", "1", "2", "3"]:
        raise RuntimeError("unknow flag_symm type " + str(flag_symm))
    return flag_symm

def read_and_check_etot_input(etot_input_path:str, kspacing:float=None, flag_symm:int=None):
    with open(etot_input_path, "r") as fp:
        lines = fp.readlines()
    
    etot_lines = lines
    # key_type = ["string_keys", "char_keys", "bool_keys", "int_keys", "float_keys"]
    key_values = {}
    for i in etot_lines[1:]:
        i = i.strip()
        key = i.split('=')[0].strip().upper()
        value = None
        if key in string_keys:
            value = i.split('=')[1]
        if key in char_keys:
            value = i.split('=')[1:]
            if len(value) > 1:
                raise Exception(" {} error, value should be char type, please check the file {}!".format(i, etot_input_path))
        elif key in bool_keys:
            value = i.split('=')[1].strip().upper()
            if key == "ENERGY_DECOMP":
                pass
            elif value not in ["T", "F"]:
                raise Exception(" {} error, value should be T or F, please check the file {}!".format(i, etot_input_path))
        elif key in int_keys:
            try:
                value = int(i.split('=')[1].strip().upper())
            except Exception:
                raise Exception(" {} error, value should be int type, please check the file {}!".format(i, etot_input_path))
        elif key in float_keys:
            try:
                value = float(i.split('=')[1].strip().upper())
            except Exception:
                raise Exception(" {} error, value should be float type, please check the file {}!".format(i, etot_input_path))
        key_values[key] = value
    return key_values, etot_lines

'''
description: 
    set etot.input content, 
    if the etot_input_file exists, read and check format, then supplement unset parameters
    else set the content according to user input json file
    
param {str} etot_input_file
param {str} atom_config
return {*}
author: wuxingxing
'''
def set_etot_input_by_file(etot_input_file:str, kspacing:float=None, flag_symm:int=None, atom_config:str=None, resource_node:list[int]=None):
    key_values, etot_lines = read_and_check_etot_input(etot_input_file, atom_config)
    # check node1 and node2 are right
    index = 0
    while index < len(etot_lines):
        if len(etot_lines[index].strip().split()) == 2:
            node1, node2 = [int(_) for _ in etot_lines[index].strip().split()]
            if node1 <= resource_node[0] and node2 <= resource_node[1]:
                pass
            else:
                raise Exception("the node1 node2 '{}' in {} file is not consistent with resource json file '{}', please check!".format(node1, node2, resource_node))
        if "in.atom".upper() in etot_lines[index].upper(): # change the in.atom
            etot_lines[index] = "in.atom = {}\n".format(os.path.basename(atom_config))
        index += 1
    key_list = list(key_values)
    # set OUT.MLMD
    if "OUT.MLMD" not in key_list:
        etot_lines.append("OUT.MLMD = T\n")
    # set OUT.WG OUT.RHO OUT.VR
    if "OUT.WG" not in key_list:
        etot_lines.append("OUT.WG = F\n")
    if "OUT.RHO" not in key_list:
        etot_lines.append("OUT.RHO = F\n")
    if "OUT.VR" not in key_list:
        etot_lines.append("OUT.VR = F\n")
    # if MP_N123 is not in etot.input file then using 'kespacing' generates it
    if "MP_N123" not in key_list:
        kspacing = PWMAT.kspacing_default if kspacing is None else kspacing
        MP_N123 = _make_kspacing_kpoints(atom_config, kspacing)
        if "FLAG_SYMM" in key_list:
            MP_N123 += str(key_values["FLAG_SYMM"])
        else:
            MP_N123 += str(flag_symm)
        etot_lines.append("MP_N123 = {}\n".format(MP_N123))
    
    if "in.atom".upper() not in key_list:   # if the in.atom not setted in etot.input file then add
        etot_lines[-1] = "in.atom = {}\n".format(os.path.basename(atom_config))

    etot_lines.append("\n")

    return "".join(etot_lines)

'''
description: 
get atom type list in the atom.config file
param {str} atom_config
return {*}
author: wuxingxing
'''
def get_atom_type_from_atom_config(atom_config:str):
    with open(atom_config, "r") as rf:
        lines = rf.readlines()
    atom_num = int(lines[0].strip().split()[0].strip())
    index = 0
    while index < len(lines):
        if "POSITION" in lines[index].upper():
            break
        index += 1
    atom_type_list = []
    for atom_line in lines[index+1:index+atom_num+1]:
       atom_name = ELEMENTTABLE_2[int(atom_line.strip().split()[0].strip())]
       if atom_name not in atom_type_list:
           atom_type_list.append(atom_name)
    return atom_type_list
    
'''
Author: WuXing wuxingxing
Date: 2023-02-28 16:48:49
'''
float_keys=[
    'Ecut',
    'Ecut2',
    'Ecut2L',
    'EcutP',
    'HSE_OMEGA',
    'HSE_ALPHA',
    'HSE_BETA',
    'NUM_ELECTRON',
    'WG_ERROR',
    'E_ERROR',
    'RHO_ERROR',
    'FERMIDE',
    'KSPACING'
    ]

int_keys=[
    'SPIN',
    'CONSTRAINT_MAG',
    'SPIN222_MAGDIR_STEPFIX',
    'DFTD3_VERSION',
    'NUM_BAND',
    'FLAG_SYMM'
    ]

bool_keys=[
    'OUT_WG',
    'IN_RHO',
    'OUT_RHO',
    'IN_VR',
    'OUT_VR',
    'IN_VEXT',
    'OUT_VATOM',
    'OUT_FORCE',
    'OUT_STRESS',
    'IN_SYMM',
    'CHARGE_DECOMP',
    'ENERGY_DECOMP',
    'IN_SOLVENT',
    'IN_NONSCF',
    'IN_OCC',
    'IN_OCC_ADIA',
    'OUT_MLMD',
    'NUM_BLOCKED_PSI',
    'OUT_RHOATOM'
    ]

char_keys=['PRECISION',
           'JOB',
           'IN_ATOM',
           'CONVERGENCE',
           'ACCURACY',
           'XCFUNCTIONAL',
           'VDW']

string_keys=['VFF_DETAIL',
           'EGG_DETAIL',
           'N123',
           'NS123',
           'N123L',
           'P123',
           'MP_N123',
           'NQ123',
           'RELAX_DETAIL',
           'MD_DETAIL',
           'MD_SPECIAL',
           'SCF_SPECIAL',
           'STRESS_CORR',
           'HSE_DETAIL',
           'RELAX_HSE',
           'IN_OCC',
           'SCF_ITER0_1',
           'SCF_ITER0_2',
           'SCF_ITER1_1',
           'ENERGY_DECOMP_SPECIAL',
           'ENERGY_DECOMP_SPECIAL1',
           'ENERGY_DECOMP_SPECIAL2',
           'ENERGY_DECOMP_COULOMB',
           ]
