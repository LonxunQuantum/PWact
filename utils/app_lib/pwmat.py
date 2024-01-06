import os
import subprocess
from utils.app_lib.poscar2lammps import p2l
from utils.constant import LAMMPSFILE, ELEMENTTABLE_2
from active_learning.user_input.param_input import SCFParam

        
    
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
        if "lattice" in ii or "Lattice" in ii or "LATTICE" in ii:
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
    atom_config_name,
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
    out_mlmd="T"
):
    script = ""
    script += "{} {}\n".format(node1, node2)
    script += "{} {}\n".format("in.atom", atom_config_name)
    script += "{} {}\n".format("ecut", ecut)
    script += "{} {}\n".format("ecut2", ecut2)
    script += "{} {}\n".format("e_error", e_error)
    script += "{} {}\n".format("rho_error", rho_error)
    script += "{} {}\n".format("out.force", out_force)
    script += "{} {}\n".format("energy_decomp", energy_decomp)
    script += "{} {}\n".format("out.stress", out_stress)
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
    script += "{} {}\n".format("scf_iter0_1", SCF_ITER0_1)
    script += "{} {}\n".format("scf_iter0_2", SCF_ITER0_2)
    if flag_symm is not None:
        MP_N123 = _make_kspacing_kpoints(atom_config, kspacing)
        MP_N123 += str(flag_symm)
    else:
        MP_N123 = _make_kspacing_kpoints(atom_config, kspacing)
    # use pwmat defualt value
    script += "{} {}\n".format("mp_n123", MP_N123)
    script += "{} {}\n".format("out.wg", out_wg)
    script += "{} {}\n".format("out.rho", out_rho)
    script += "{} {}\n".format("out.mlmd", out_mlmd)
    return script

def _update_input_dict(input_dict_, user_dict):
    if user_dict is None:
        return input_dict_
    input_dict = input_dict_
    for ii in user_dict:
        input_dict[ii] = user_dict[ii]
    return input_dict


def write_input_dict(input_dict):
    lines = []
    for key in input_dict:
        if type(input_dict[key]) == bool:
            if input_dict[key]:
                rs = "T"
            else:
                rs = "F"
        else:
            rs = str(input_dict[key])
        lines.append("%s=%s" % (key, rs))
    return "\n".join(lines)


def _make_smearing(icmix=None, smearing = None, sigma = None):
    if icmix == None:
        if smearing == None:
            if sigma == None:
                return None, None, None
            else:
                return None, None, sigma
        else:
            if sigma == None:
                return None, smearing, None
            else:
                return None, smearing, sigma
    else:
        if smearing == None:
            if sigma == None:
                return icmix, None, None
            else:
                return icmix, None, sigma
        else:
            if sigma == None:
                return icmix, smearing, None
            else:
                return icmix, smearing, sigma
    

def _make_flag_symm(flag_symm = None):
    if flag_symm == None:
        return None
    if flag_symm == "NONE":
        flag_symm = None
    elif str(flag_symm) not in [None, "0", "1", "2", "3"]:
        raise RuntimeError("unknow flag_symm type " + str(flag_symm))
    return flag_symm


def make_pwmat_input_user_dict(fp_params):
    node1 = fp_params["node1"]
    node2 = fp_params["node2"]
    atom_config = fp_params["in.atom"]
    ecut = fp_params["ecut"]
    e_error = fp_params["e_error"]
    rho_error = fp_params["rho_error"]
    kspacing = fp_params["kspacing"]
    if "user_pwmat_params" in fp_params:
        user_dict = fp_params["user_pwmat_params"]
    else:
        user_dict = None
    icmix, smearing, sigma = _make_smearing(fp_params)
    flag_symm = _make_flag_symm(fp_params)
    input_dict = make_pwmat_input_dict(
        node1,
        node2,
        atom_config,
        ecut,
        e_error,
        rho_error,
        icmix=icmix,
        smearing=smearing,
        sigma=sigma,
        kspacing=kspacing,
        flag_symm=flag_symm,
    )
    input_dict = _update_input_dict(input_dict, user_dict)
    input = write_input_dict(input_dict)
    return input

def read_and_check_etot_input(etot_input_path):
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
            if value not in ["T", "F"]:
                raise Exception(" {} error, value should be T or F, please check the file {}!".format(i, etot_input_path))
        elif key in int_keys:
            try:
                value = int(i.split('=')[1].strip().upper())
            except:
                raise Exception(" {} error, value should be int type, please check the file {}!".format(i, etot_input_path))
        elif key in float_keys:
            try:
                value = float(i.split('=')[1].strip().upper())
            except:
                raise Exception(" {} error, value should be float type, please check the file {}!".format(i, etot_input_path))
        key_values[key] = value
    key_list = key_values.keys()
    # check necessary keys:
    if "MP_N123" not in key_list and "KSPACING" not in key_list:
        raise Exception(" MP_N123 or KSPACING must be set, please check the file {}!".format(etot_input_path))
    
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
def set_etot_input_by_file(etot_input_file:str, atom_config:str):
    key_values, etot_lines = read_and_check_etot_input(etot_input_file)
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
        MP_N123 = _make_kspacing_kpoints(atom_config, key_values["KSPACING"])
        if "FLAG_SYMM" in key_list:
            MP_N123 += str(key_values["FLAG_SYMM"])
        etot_lines.append("MP_N123 = {}\n".format(MP_N123))
    etot_lines.append("\n")
    
    return "".join(etot_lines)

def set_etot_input_content(etot_input_file:str=None, atom_config:str=None, scfparam:SCFParam=None):
    if etot_input_file is not None and os.path.exists(etot_input_file):
        etot_input_content = set_etot_input_by_file(etot_input_file, atom_config)
    else:

        icmix, smearing, sigma = _make_smearing(scfparam.icmix, scfparam.smearing, scfparam.sigma)
        flag_symm = _make_flag_symm(scfparam.flag_symm)
        
        etot_input_content = make_pwmat_input_dict(
            node1 = scfparam.node1,
            node2 = scfparam.node2,
            atom_config_name = scfparam.atom_config_name,
            ecut = scfparam.ecut,
            ecut2 = scfparam.ecut2,
            e_error = scfparam.e_error,
            rho_error = scfparam.rho_error,
            out_force = scfparam.out_force,
            energy_decomp = scfparam.energy_decomp,
            out_stress = scfparam.out_stress,
            icmix = icmix,
            smearing = smearing,
            sigma = sigma,
            kspacing = scfparam.kspacing,
            flag_symm = flag_symm,
            out_wg = scfparam.out_wg,
            out_rho = scfparam.out_rho,
            out_mlmd = scfparam.out_mlmd
        )
    return etot_input_content

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
        if "POSITION" in line.upper():
            break
        index += 1
    atom_type_list = []
    for atom_line in lines[index:index+atom_num]:
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
