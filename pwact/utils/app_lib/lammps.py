#!/usr/bin/env python3
import random
import os
import numpy as np

from pwact.utils.constant import FORCEFILED
from pwdata.calculators.const import ELEMENTMASSTABLE
def _sample_sphere():
    while True:
        vv = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
        vn = np.linalg.norm(vv)
        if vn < 0.2:
            continue
        return vv / vn

def make_pair_style(md_type, forcefiled, atom_type:list[int], dump_info:str):
    pair_style = ""
    
    if md_type == FORCEFILED.fortran_lmps:
        raise Exception("the fortran in.lammps not relized!")
        pass
    
    elif md_type == FORCEFILED.libtorch_lmps:
        pair_names = ""
        for fi in forcefiled:
            pair_names += "{} ".format(os.path.basename(fi))
        pair_style = "pair_style   pwmlff   {} {} {}\n".format(len(forcefiled), pair_names, dump_info)
    atom_names = " ".join(map(str, atom_type))
    pair_style += "pair_coeff       * * {}\n".format(atom_names)
    return pair_style

# def make_mass(mass):
#     if mass is None:
#         return ""
    
#     mass_str = "\n"
#     if isinstance(mass, str):
#         mass_str += "mass       {}\n".format(mass)
#     else:
#         for m in mass:
#             mass_str += "mass       {}\n".format(m)
#     return mass_str

def make_mass(atom_type:list):
    if isinstance(atom_type, str):
        atom_type = [atom_type]
    if atom_type is None:
        return ""
    mass_str = "\n"
    
    for i, atom in enumerate(atom_type):
        mass_str += "mass   {}    {}\n".format(i+1, ELEMENTMASSTABLE[atom])
    return mass_str

def set_ensemble(ensemble):
    pass

def make_lammps_input(
    md_file,
    md_type,
    forcefiled,
    atom_type,
    ensemble,
    nsteps,
    dt,
    neigh_modify,
    trj_freq,
    mass,
    temp,
    tau_t,
    press,
    tau_p,    
    boundary, #true is 'p p p', false is 'f f f', default is true
    merge_traj,
    max_seed=100000,
    restart=0,
    model_deviation_file = "model_deviation.out"
):
    md_script = ""
    md_script += "variable        NSTEPS          equal %d\n" % nsteps
    md_script += "variable        THERMO_FREQ     equal %d\n" % trj_freq
    md_script += "variable        DUMP_FREQ       equal %d\n" % trj_freq
    md_script += "variable        restart         equal %d\n" % restart

    md_script += "variable        TEMP            equal %f\n" % temp
    # if ele_temp_f is not None:
    #     md_script += "variable    ELE_TEMP        equal %f\n" % ele_temp_f
    # if ele_temp_a is not None:
    #     md_script += "variable    ELE_TEMP        equal %f\n" % ele_temp_a
    if press is not None:
        md_script += "variable        PRESS           equal %f\n" % press
    md_script += "variable        TAU_T           equal %f\n" % tau_t
    if tau_p is not None:
        md_script += "variable        TAU_P           equal %f\n" % tau_p
    md_script += "\n"
    
    md_script += "units           metal\n"

    if boundary:
        md_script += "boundary        p p p\n"
    else:
        md_script += "boundary        f f f\n"
    
    md_script += "atom_style      atomic\n"
    md_script += "\n"
    md_script += "neighbor        1.0 bin\n"
    if neigh_modify is not None:
        md_script += "neigh_modify    delay %d\n" % neigh_modify
    md_script += "\n"
    md_script += "box              tilt large\n"
    
    md_script += (
        'if "${restart} > 0" then "read_restart lmps.restart.*" else "read_data %s"\n'
        % md_file
    )
    md_script += "change_box       all triclinic\n"
    
    md_script += make_mass(atom_type)
    dump_info = "out_freq ${{DUMP_FREQ}} out_file {} ".format(model_deviation_file)
    md_script += make_pair_style(md_type, forcefiled, atom_type, dump_info)
    #put_freq ${freq} out_file error
    
    md_script += "\n"
    md_script += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
    md_script += "thermo          ${THERMO_FREQ}\n"
    
    if merge_traj is True:
        md_script += "dump            1 all custom ${DUMP_FREQ} all.lammpstrj id type x y z fx fy fz\n"
        md_script += 'if "${restart} > 0" then "dump_modify     1 append yes"\n'
    else:
        md_script += "dump            1 all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z fx fy fz\n"
    
    md_script += "restart         10000 lmps.restart\n"
    md_script += "\n"
    
    md_script += 'if "${restart} == 0" then "velocity        all create ${TEMP} %d"' % (
            random.randrange(max_seed - 1) + 1
        )
        
    md_script += "\n"
    
    if ensemble.split("-")[0] == "npt":
        assert press is not None
        if not boundary:
            raise RuntimeError("ensemble %s is conflicting with boundary" % ensemble)
    if ensemble == "npt" or ensemble == "npt-i" or ensemble == "npt-iso":
        md_script += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRESS} ${PRESS} ${TAU_P}\n"
    elif ensemble == "npt-a" or ensemble == "npt-aniso":
        md_script += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRESS} ${PRESS} ${TAU_P}\n"
    elif ensemble == "npt-t" or ensemble == "npt-tri":
        md_script += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRESS} ${PRESS} ${TAU_P}\n"
    elif ensemble == "nvt":
        md_script += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
    elif ensemble == "nve":
        md_script += "fix             1 all nve\n"
    else:
        raise RuntimeError("unknown emsemble " + ensemble)
    
    if not boundary:
        md_script += "velocity        all zero linear\n"
        md_script += "fix             fm all momentum 1 linear 1 1 1\n"
    md_script += "\n"
    md_script += "timestep        %f\n" % dt
    md_script += "run             ${NSTEPS} upto\n"
    
    return md_script

def get_dumped_forces(file_name):
    with open(file_name) as fp:
        lines = fp.read().split("\n")
    natoms = None
    for idx, ii in enumerate(lines):
        if "ITEM: NUMBER OF ATOMS" in ii:
            natoms = int(lines[idx + 1])
            break
    if natoms is None:
        raise RuntimeError(
            "wrong dump file format, cannot find number of atoms", file_name
        )
    idfx = None
    for idx, ii in enumerate(lines):
        if "ITEM: ATOMS" in ii:
            keys = ii
            keys = keys.replace("ITEM: ATOMS", "")
            keys = keys.split()
            idfx = keys.index("fx")
            idfy = keys.index("fy")
            idfz = keys.index("fz")
            break
    if idfx is None:
        raise RuntimeError("wrong dump file format, cannot find dump keys", file_name)
    ret = []
    for ii in range(idx + 1, idx + natoms + 1):
        words = lines[ii].split()
        ret.append([float(words[ii]) for ii in [idfx, idfy, idfz]])
    ret = np.array(ret)
    return ret


def get_all_dumped_forces(file_name):
    with open(file_name) as fp:
        lines = fp.read().split("\n")

    ret = []
    exist_natoms = False
    exist_atoms = False

    for idx, ii in enumerate(lines):

        if "ITEM: NUMBER OF ATOMS" in ii:
            natoms = int(lines[idx + 1])
            exist_natoms = True

        if "ITEM: ATOMS" in ii:
            keys = ii
            keys = keys.replace("ITEM: ATOMS", "")
            keys = keys.split()
            idfx = keys.index("fx")
            idfy = keys.index("fy")
            idfz = keys.index("fz")
            exist_atoms = True

            single_traj = []
            for jj in range(idx + 1, idx + natoms + 1):
                words = lines[jj].split()
                single_traj.append([float(words[jj]) for jj in [idfx, idfy, idfz]])
            single_traj = np.array(single_traj)
            ret.append(single_traj)

    if exist_natoms is False:
        raise RuntimeError(
            "wrong dump file format, cannot find number of atoms", file_name
        )
    if exist_atoms is False:
        raise RuntimeError("wrong dump file format, cannot find dump keys", file_name)
    return ret


