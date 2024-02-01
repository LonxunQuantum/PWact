from utils.constant import DFT_STYLE, VASP, PWMAT, LAMMPS,\
    ELEMENTTABLE_2, ELEMENTTABLE
from utils.file_operation import link_file, merge_files_to_one, write_to_file, copy_file, mv_file
from utils.app_lib.pwmat import set_etot_input_by_file, atom_config_to_poscar, poscar_to_atom_config, poscar_to_lammps_in
import os

'''
description: 
    this script for pwmat, vasp, ... common operation
author: wuxingxing
'''

'''
description: 
    link the source config to target_dir
    if the target is poscar:
        if the source is atom.config:
            convert atom.config to poscar
        else the target is pwmat:
            link the source atom.config to target dir

    else if the target is atom.config:
        if the source is poscar:
            convert poscar to atom.config
        else the target is pwmat:
            link the source atom.config to target dir

    elif if the target is lammps format:
        if the source is poscar:
            convert poscar to lammps format
        else if the source is atom.config:
            convert the source atom.config to poscar then convert the poscar to lammps in
        else the source is lammps format:
            link to target dir

param {str} source_config
param {str} config_format
param {str} target_dir
param {str} dft_style
return {*}
author: wuxingxing
'''
def link_structure(source_config:str, config_format:str, target_dir:str, dft_style:str):
    
    if dft_style == DFT_STYLE.pwmat:
        if config_format == DFT_STYLE.vasp:
            # the input config is vasp poscar, covert it to atom.confie file then mv to target dir
            target_file = poscar_to_atom_config(poscar=source_config, 
                save_dir=target_dir, 
                save_name="poscar_to_atom.config")
            target_config = os.path.join(target_dir, PWMAT.atom_config)
            mv_file(target_file, target_config)
        else:
            # the input config is pwmat atom.config, then link to target dir
            target_config = os.path.join(target_dir, os.path.basename(source_config))
            link_file(source_config, target_config)

    elif dft_style == DFT_STYLE.vasp:
        if config_format == DFT_STYLE.pwmat:
            # the input config is pwmat config, convert it to poscar file and mv to target dir
            target_config = atom_config_to_poscar(atom_config=source_config, save_dir=target_dir)
        else:
            # the input config is vasp poscar, link to target dir
            target_config = os.path.join(target_dir, os.path.basename(source_config))
            link_file(source_config, target_config)
    
    elif dft_style == DFT_STYLE.lammps:
        if config_format == DFT_STYLE.pwmat:
            atom_config_to_poscar(atom_config=source_config, save_dir=target_dir)
        elif config_format == DFT_STYLE.vasp:
            copy_file(source_config, os.path.join(target_dir, VASP.poscar))
        target_config = poscar_to_lammps_in(target_dir)
        
    return target_config

'''
description: 
    link the pseudo files to target dir
    for pwmat pseduo files, link each file to target dir
    for vasp pseduo files, merge files to one potcar file according to atom_order, then write to target dir
param {list} pseudo_list
param {str} target_dir
param {list} atom_order
param {str} dft_style
return {*}
author: wuxingxing
'''
def link_pseudo_by_atom(pseudo_list:list, target_dir:str, atom_order:list[str], dft_style:str):
    pseudo_find = []
    if dft_style == DFT_STYLE.pwmat:
        for atom_name in atom_order:
            for pseudo in pseudo_list:
                pseudo_path = pseudo[0]
                pseudo_name = os.path.basename(pseudo_path)
                atom_type = pseudo[1]
                if atom_name == atom_type:
                    link_file(pseudo_path, os.path.join(target_dir, pseudo_name))
                    pseudo_find.append(pseudo_path)
                    break
        assert len(pseudo_find) == len(atom_order), "the pwmat pseudo files {} not same as atom type '{}'".format(pseudo_find, atom_order)
    
    elif dft_style == DFT_STYLE.vasp:
        # merge file to where? to save dir
        for atom_name in atom_order:
            for pseudo in pseudo_list:
                pseudo_path = pseudo[0]
                atom_type = pseudo[1]
                if atom_name == atom_type:
                    pseudo_find.append(pseudo_path)
        assert len(pseudo_find) == len(atom_order), "the vasp pseudo files {} not same as atom type '{}'".format(pseudo_find, atom_order)
        target_pseudo_file = os.path.join(target_dir, VASP.potcar)
        merge_files_to_one(pseudo_find, target_pseudo_file)
    else:
        pass

    return [os.path.basename(_) for _ in pseudo_find] # this return is for pwmat etot.input for IN.PSP set

'''
description: 
    get atom type in pseudo file
param {str} pseduo_path
return {*}
author: wuxingxing
'''
def get_vasp_pseudo_atom_type(pseduo_path:str):
    with open(pseduo_path, "r") as rf:
        line = rf.readline()
    atom_type = line.split()[1].split('_')[0]
    return atom_type

'''
description: 
    get atom type
param {str} config
param {str} dft_style, it could be vasp poscar file or pwmat atom.config file
return {*}
author: wuxingxing
'''
def get_atom_type(config:str, dft_style:str):
    if dft_style == DFT_STYLE.pwmat:
        return get_atom_type_from_atom_config(config)
    elif dft_style == DFT_STYLE.vasp:
        return get_atom_type_from_poscar(config)
    else:
        pass

'''
description: 
    read the atom type order in poscar
    for poscar file, the 5-th line is its atom type and order
param {str} config
return {*}
author: wuxingxing
'''
def get_atom_type_from_poscar(config:str):
    with open(config, "r") as rf:
        lines = rf.readlines()
    atom_type_list = lines[5].split()
    atomic_number_list = []
    for atom in atom_type_list:
        atomic_number_list.append(ELEMENTTABLE[atom])
    return atom_type_list, atomic_number_list

'''
description: 
get atom type list in the atom.config file, the order is same as atom.config
param {str} atom_config
return {*} atomic type name list, atomic number list
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
    atomic_number_list = []
    for atom_line in lines[index+1:index+atom_num+1]:
        atomic_number = int(atom_line.strip().split()[0].strip())
        if atomic_number not in atomic_number_list:
            atom_type_list.append(ELEMENTTABLE_2[atomic_number])
            atomic_number_list.append(atomic_number)
    return atom_type_list, atomic_number_list

'''
description: 
    set input script, for pwmat is etot.input file, for vasp is INCAR file
return {*}
author: wuxingxing
'''
def set_input_script(            
    input_file:str=None,    # pwmat is etot.input, vasp is INCAR
    config:str=None, 
    kspacing:float=None, 
    flag_symm:int=None, 
    dft_style:str=None,
    save_dir:str=None,
    pseudo_names:list[str]=None
    ):
    if dft_style == DFT_STYLE.pwmat:
        target_file = os.path.join(save_dir, PWMAT.etot_input)
        script = set_etot_input_by_file(    #do pwmat etot.input content check
            etot_input_file=input_file, 
            atom_config=config, 
            kspacing=kspacing, 
            flag_symm=flag_symm, 
            pseudo_names=pseudo_names
            )
        write_to_file(target_file, script, "w")
    elif dft_style == DFT_STYLE.vasp:
        #do pwmat INCAR content check, not realized yet.
        target_file = os.path.join(save_dir, VASP.incar)
        copy_file(input_file, target_file)
    else:
        pass