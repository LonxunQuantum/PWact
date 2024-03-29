import os
import numpy as np
import subprocess
from pwact.utils.constant import PWMAT, VASP
from pwact.utils.file_operation import del_file, copy_file
# '''
# description: 
#     lammps dump file to poscar format or pwmat format
# param {str} dump_file
# param {str} save_file
# param {list} type_map
# param {bool} unwrap
# return {*}
# author: wuxingxing
# '''
# def lammps_dump_to_config(dump_file:str, save_file:str, type_map:list[str], unwrap:bool=False, dft_style:str=None):
#     system = dp_system(dump_file, type_map=type_map, begin=0, step=1, unwrap=False, fmt = 'lammps/dump')
#     tmp_poscar = os.path.join(os.path.dirname(save_file), "tmp_psocar")
#     system.to_poscar(tmp_poscar, frame_idx=0)
#     if dft_style == DFT_STYLE.pwmat:
#         target_file = poscar_to_atom_config(
#             poscar=tmp_poscar, 
#             save_dir=os.path.dirname(save_file), 
#             save_name=os.path.basename(save_file)
#             )
#         del_file(tmp_poscar)
#     elif dft_style == DFT_STYLE.vasp:
#         target_file = os.path.join(os.path.dirname(save_file), VASP.poscar)
#         copy_file(tmp_poscar, target_file)
#         del_file(tmp_poscar)
#     return target_file
        


'''
description: 
    1. copy the source poscar to save_dir/tmp_poscar
    2. convert the tmp_poscar to atom.config
    3. change the atom.config file name to save_name
    4. delete the tmp_poscar file
param {str} poscar
param {str} save_dir
param {str} save_name
return {*}
author: wuxingxing
'''
def poscar_to_atom_config(poscar:str, save_dir:str, save_name:str=None):
    tmp_poscar = os.path.join(save_dir, "tmp_poscar")
    copy_file(poscar, tmp_poscar)
    cwd = os.getcwd()
    os.chdir(save_dir)
    subprocess.run(["poscar2config.x {} > /dev/null".format(os.path.basename(tmp_poscar))], shell = True)
    if save_name is not None:
        subprocess.run(["mv {} {} > /dev/null".format(PWMAT.atom_config, save_name)], shell = True)
        target_file = os.path.join(save_dir, save_name)
    else:
        target_file = os.path.join(save_dir, PWMAT.atom_config)
    os.chdir(cwd)
    del_file(tmp_poscar)
    return target_file

'''
description:
    1. copy the source atom.config to save_dir/tmp_atom.config
    2. convert the tmp_atom.config to POSCAR
    3. change the POSCAR file name to save_name
    4. delete the tmp_atom.config file
param {str} atom_config
param {str} save_dir
param {str} save_name
return {*}
author: wuxingxing
'''
def atom_config_to_poscar(atom_config:str, save_dir:str, save_name:str=None):
    tmp_atom_config = os.path.join(save_dir, "tmp_atom.config")
    copy_file(atom_config, tmp_atom_config)
    cwd = os.getcwd()
    os.chdir(save_dir)
    subprocess.run(["config2poscar.x {} > /dev/null".format(os.path.basename(tmp_atom_config))], shell = True)
    # remove the black space in ' Selective Dynamics' line in POSCAR file
    subprocess.run(["sed -i '/^ *Selective Dynamics/s/^ *//' {} > /dev/null".format(VASP.poscar)], shell = True)
    # remame the poscar file
    if save_name is not None:
        subprocess.run(["mv {} {} > /dev/null".format(VASP.poscar, save_name)], shell = True)
        target_file = os.path.join(save_dir, save_name)
    else:
        target_file = os.path.join(save_dir, VASP.poscar)
    os.chdir(cwd)
    del_file(tmp_atom_config)# delete temp file
    return target_file

def _reciprocal_box(box):
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    return rbox

# def _make_pwmat_kp_mp(kpoints):
#     ret = ""
#     ret += "%d %d %d 0 0 0 " % (kpoints[0], kpoints[1], kpoints[2])
#     return ret

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
        round(2 * np.pi * np.linalg.norm(ii) / kspacing) for ii in rbox
    ]
    kpoints[0] = 1 if kpoints[0] == 0 else kpoints[0]
    kpoints[1] = 1 if kpoints[1] == 0 else kpoints[1]
    kpoints[2] = 1 if kpoints[2] == 0 else kpoints[2]
    ret = ""
    ret += "%d %d %d 0 0 0 " % (kpoints[0], kpoints[1], kpoints[2])
    return ret
    # ret = _make_pwmat_kp_mp(kpoints)

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

def read_and_check_etot_input(etot_input_path:str):
    with open(etot_input_path, "r") as fp:
        lines = fp.readlines()
    
    etot_lines = lines
    # key_type = ["string_keys", "char_keys", "bool_keys", "int_keys", "float_keys"]
    key_values = {}
    for i in etot_lines[1:]:
        i = i.strip()
        if "#" in i[:2]:
            continue
        key = i.split('=')[0].strip().upper()
        value = None
        if key in string_keys:
            value = i.split('=')[1]
        if key in char_keys:
            value = i.split('=')[1:]
            if len(value) > 1:
                raise Exception(" {} error, value should be char type, please check the file {}!".format(i, etot_input_path))
        elif key in bool_keys:#USE_DFTB
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
    check mp_123 and kspacing for scf etot.input file
param {*} self
param {str} etot_input
param {None} kspacing
return {*}
author: wuxingxing
'''
def check_kspacing_mp_123(etot_input:str, kspacing:None):
    key_values, etot_lines = read_and_check_etot_input(etot_input)
    if kspacing is not None and "MP_123" in key_values:
        error_info = "The 'kspacing' in DFT/input/{} dict and 'MP_123' in ethot.input cannot coexist.\n".format(os.path.basename(etot_input))
        error_info += "If 'MP_123' is not indicated in DFT/input/{}, we will use 'kspacing' param to generate the 'MP_123' parameter\n".format(os.path.basename(etot_input))
        raise Exception(error_info)
        

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
def set_etot_input_by_file(
    etot_input_file:str, 
    kspacing:float=None, 
    flag_symm:int=None, 
    atom_config:str=None, 
    pseudo_names:list[str]=None,
    is_scf = False, # if True, job is scf, and 'OUT.MLMD = T' to etot.input
    is_skf_file = False  # if True, set in.skf to etot.input file
    ):
    key_values, etot_lines = read_and_check_etot_input(etot_input_file)
    is_skf = False
    if "USE_DFTB" in key_values.keys() and key_values["USE_DFTB"] is not None and key_values["USE_DFTB"] == "T":
        if key_values["DFTB_DETAIL"].replace(",", " ").split()[0] != "3": # not chardb
            is_skf = True
    index = 0
    new_etot_lines = []
    while index < len(etot_lines):
        # remove the in.atom, in.skf in.psp* in etot.input file if exists
        if "in.atom".upper() in etot_lines[index].upper():
            # etot_lines[index].remove(etot_lines[index])
            pass
        elif "in.skf".upper() in etot_lines[index].upper():
            # etot_lines[index].remove(etot_lines[index])
            pass
        elif "IN.PSP" in etot_lines[index].upper(): # to avoid the new_etot_lines add 'IN.PSP' and 'in.skf' 'in.atom' in etot_lines
            pass
            # etot_lines.remove(etot_lines[index])
        else:
            new_etot_lines.append(etot_lines[index])
        index += 1
    new_etot_lines.append("\nIN.ATOM = {}\n".format(os.path.basename(atom_config)))
    # if dftb and need in_skf
    if is_skf and is_skf_file:
        new_etot_lines.append("IN.SKF = ./{}/\n".format(PWMAT.in_skf))
    # is not for dftb, reset the IN.PSP
    if "USE_DFTB" not in key_values.keys() or key_values["USE_DFTB"] is None and key_values["USE_DFTB"] == "F":
        for pseudo_i, pseudo in enumerate(pseudo_names):
            new_etot_lines.append("IN.PSP{} = {}\n".format(pseudo_i + 1, pseudo))
    key_list = list(key_values)
    # set OUT.MLMD
    if "OUT.MLMD" not in key_list:
        if is_scf:
            new_etot_lines.append("OUT.MLMD = T\n")
    # # set OUT.WG OUT.RHO OUT.VR
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
        new_etot_lines.append("MP_N123 = {}\n".format(MP_N123))

    new_etot_lines.append("\n")
    
    return "".join(new_etot_lines)

def is_alive_atomic_energy(movement_list:list):
    if len(movement_list) < 1:
        return False
    # Declare is_real_Ep as a global variable
    command = 'grep Atomic-Energy ' + movement_list[0] + ' | head -n 1'
    print('running-shell-command: ' + command)
    result = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8', shell=True)
    if 'Atomic-Energy' in result.stdout:
        alive_atomic_energy = True
    else:
        alive_atomic_energy = False
    return alive_atomic_energy
    
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
    'KSPACING',
    "SYMM_PREC" #DFTB param
    ]

int_keys=[
    'SPIN',
    'CONSTRAINT_MAG',
    'SPIN222_MAGDIR_STEPFIX',
    'DFTD3_VERSION',
    'NUM_BAND',
    'FLAG_SYMM',
    "NUM_GPU", #DFTB param
    "RANDOM_SEED"#DFTB param
    ]

bool_keys=[
    'OUT.WG',
    'IN.RHO',
    'OUT.RHO',
    'IN.VR',
    'OUT.VR',
    'IN.VEXT',
    'OUT.VATOM',
    'OUT.FORCE',
    'OUT.STRESS',
    'IN.SYMM',
    'CHARGE_DECOMP',
    'ENERGY_DECOMP',
    'IN.SOLVENT',
    'IN.NONSCF',
    'IN.OCC',
    'IN.OCC_ADIA',
    'OUT.MLMD',
    'NUM_BLOCKED_PSI',
    'OUT.RHOATOM',
    "USE_DFTB"
    ]

char_keys=['PRECISION',
           'JOB',
           'IN.ATOM',
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
           "DFTB_DETAIL",#DFTB param
           'MD_SPECIAL',
           'SCF_SPECIAL',
           'STRESS_CORR',
           'HSE_DETAIL',
           'RELAX_HSE',
           'IN.OCC',
           'SCF_ITER0_1',
           'SCF_ITER0_2',
           'SCF_ITER1_1',
           'ENERGY_DECOMP_SPECIAL',
           'ENERGY_DECOMP_SPECIAL1',
           'ENERGY_DECOMP_SPECIAL2',
           'ENERGY_DECOMP_COULOMB',
           ]
