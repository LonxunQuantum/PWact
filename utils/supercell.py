import os
import argparse
from utils.app_lib.pwmat import poscar_to_atom_config, atom_config_to_poscar
from utils.constant import PWMAT, DFT_STYLE
from utils.file_operation import copy_file, del_file
from ase.io.vasp import read_vasp, write_vasp
from ase.build import make_supercell
from ase import Atoms

def check_dimension(d1:int, d2:int, d3:int):
    if d1 == 0 or d2 == 0 or d3 == 0:
        raise Exception("Error! The input super cell dimension '{} {} {}' is error! No value should be 0!".format(d1, d2, d3))

'''
description: 
    atom.config super cell:
    do supper cell according to dimension, if the savepath is None, return the content after supercell
param {list} dims
param {str} filename
param {str} savepath
return {*}
author: wuxingxing
'''
def super_cell(dims:list[int], filename:str, savepath:str=None):
    n1=dims[0]
    n2=dims[1]
    n3=dims[2]
    check_dimension(n1, n2, n3)
    n123 = n1 * n2 * n3
    text = []
    lat = [[0.0,0.0,0.0] for col in range(3)]
    new_lat = [[0.0,0.0,0.0] for col in range(3)]
    fin = open(filename)
    text = fin.readlines()
    fin.close()

    num_atom = int(text[0].split()[0])
    lat[0] = [float(col) for col in text[2].split()[0:3]]
    lat[1] = [float(col) for col in text[3].split()[0:3]]
    lat[2] = [float(col) for col in text[4].split()[0:3]]

    atom_type = [0 for row in range(num_atom)]
    x_frac = [ [0.0, 0.0, 0.0] for row in range(num_atom)]
    imov_at = [[0, 0, 0] for row in range(num_atom)]
    #initial new variables
    new_num_atom = num_atom * n123
    new_x_frac = [ [0.0, 0.0, 0.0] for row in range(new_num_atom)]
    new_imov = [ [0, 0, 0] for row in range(new_num_atom)]
    new_type = [ 0 for col in range(new_num_atom)]
    for i in range(num_atom):
        atom_type[i] = int(text[i+6].split()[0])
        x_frac[i] = [float(col) for col in text[i+6].split()[1:4]]
        imov_at[i] = [int(col) for col in text[i+6].split()[4:7]]
        
    for i in range(3):
        new_lat[0][i] = lat[0][i] * n1
        new_lat[1][i] = lat[1][i] * n2
        new_lat[2][i] = lat[2][i] * n3

    for iatom in range(num_atom):
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    num_tmp = iatom*n123+i*n2*n3+j*n3+k
                    new_type[num_tmp] = atom_type[iatom]
                    new_imov[num_tmp] = imov_at[iatom]
                    new_x_frac[num_tmp][0] = x_frac[iatom][0] / float(n1) + float(i) / float(n1)
                    new_x_frac[num_tmp][1] = x_frac[iatom][1] / float(n2) + float(j) / float(n2)
                    new_x_frac[num_tmp][2] = x_frac[iatom][2] / float(n3) + float(k) / float(n3)

    lattic_index = 0
    super_cell_content = []
    super_cell_content.append("%d\n" % new_num_atom)
    super_cell_content.append("Lattice vector\n")
    lattic_index = len(super_cell_content)+1
    super_cell_content.append("%16.10f %16.10f %16.10f\n" % (new_lat[0][0], new_lat[0][1], new_lat[0][2]))
    super_cell_content.append("%16.10f %16.10f %16.10f\n" % (new_lat[1][0], new_lat[1][1], new_lat[1][2]))
    super_cell_content.append("%16.10f %16.10f %16.10f\n" % (new_lat[2][0], new_lat[2][1], new_lat[2][2]))
    super_cell_content.append("Position, move_x, mov_y, move_z\n")
    for i in range(new_num_atom):
        super_cell_content.append("%4d %16.10f %16.10f %16.10f %2d %2d %2d\n" % (new_type[i], \
    new_x_frac[i][0], new_x_frac[i][1], new_x_frac[i][2], \
    new_imov[i][0], new_imov[i][1], new_imov[i][2]))
    
    if savepath is not None:
        with open(savepath, 'w') as wf:
            wf.writelines(super_cell_content)
    return super_cell_content, lattic_index

def read_config_by_ase(source_config:str, source_config_format:str, save_dir:str):
    if source_config_format == DFT_STYLE.pwmat:
        # convert to poscar
        tmp_poscar_path = atom_config_to_poscar(source_config, save_dir, save_name="temp_poscar")
        structure = read_vasp(tmp_poscar_path)
        del_file(tmp_poscar_path)

    elif source_config_format == DFT_STYLE.vasp:
        structure = read_vasp(source_config)
    else:
        pass
    return structure

def write_config_by_ase(structure, save_config:str, save_format:str):
    if save_format == DFT_STYLE.pwmat:
        temp_poscar = os.path.join(os.path.dirname(save_config), "temp_save_poscar")
        write_vasp(temp_poscar, structure, direct=True, sort=True)
        poscar_to_atom_config(temp_poscar, os.path.dirname(save_config), os.path.basename(save_config))
        #delete tmp file
        del_file(temp_poscar)
    elif save_format == DFT_STYLE.vasp:
        write_vasp(save_config, structure, direct=True, sort=True)

'''
description: 
    use ase do super cell for atom.config file
    1. convert atom.config to poscar
    2. do super cell
    3. resort by atom type
param {str} atom_config
return {*}
author: wuxingxing
'''
def super_cell_ase(super_cell:list, source_config:str, save_config:str, dft_stype:str, source_config_format:str):
    structure = read_config_by_ase(source_config, source_config_format, os.path.dirname(save_config))
    supercell = make_supercell(structure, super_cell)
    write_config_by_ase(supercell, save_config, dft_stype)
    return save_config

'''
description: 
    functions may move to vasp and pwmat module
param {str} input_scale_config
param {float} scale
param {str} save_scale_config
param {str} dft_style
return {*}
author: wuxingxing
'''
def scale_config(input_scale_config:str, scale:float, save_scale_config:str=None, dft_style:str=None, source_config_format:str=None):
    structure = read_config_by_ase(input_scale_config, source_config_format, os.path.dirname(save_scale_config))
    #do scale
    atomic_positions = structure.get_positions()
    lattice_parameters = structure.get_cell()
    scaled_atomic_positions = atomic_positions * scale
    scaled_lattice_parameters = lattice_parameters * scale
    new_structure = Atoms(structure.get_chemical_symbols(), scaled_atomic_positions, cell=scaled_lattice_parameters)
    # svae
    write_config_by_ase(new_structure, save_scale_config, dft_style)

'''
description: 
these 2 funcitons not used
param {str} source_scale_config
param {str} source_config_format
param {str} dft_stype
param {float} scale
param {str} save_scale_config
return {*}
author: wuxingxing
'''
def scale_config_vasp(source_scale_config:str, source_config_format:str, dft_stype:str, scale:float, save_scale_config:str=None):
    poscar = read_config_by_ase(source_scale_config, source_config_format)
    atomic_positions = poscar.get_positions()
    lattice_parameters = poscar.get_cell()
    scaled_atomic_positions = atomic_positions * 0.99
    scaled_lattice_parameters = lattice_parameters * 0.99
    new_poscar = Atoms(poscar.get_chemical_symbols(), scaled_atomic_positions, cell=scaled_lattice_parameters)
    write_config_by_ase(new_poscar, save_scale_config, dft_stype)

def scale_config_pwmat(input_scale_config:str, scale:float, save_scale_config:str=None):
    with open(input_scale_config, 'r') as rf:
        text = rf.readlines()
    lattic_index = 0
    while lattic_index < len(text):
        if "LATTIC" in text[lattic_index].upper():
            lattic_index +=1
            break
        lattic_index += 1
    lat = [[0.0,0.0,0.0] for col in range(3)]
    lat[0] = [float(col) for col in text[lattic_index].split()[0:3]]
    lat[1] = [float(col) for col in text[lattic_index+1].split()[0:3]]
    lat[2] = [float(col) for col in text[lattic_index+2].split()[0:3]]
    for i in range(3):
        lat[0][i] = lat[0][i] * scale
        lat[1][i] = lat[1][i] * scale
        lat[2][i] = lat[2][i] * scale
    text[lattic_index] = "\t{}\t{}\t{}\n".format(lat[0][0], lat[0][1], lat[0][2])
    text[lattic_index+1] = "\t{}\t{}\t{}\n".format(lat[1][0], lat[1][1], lat[1][2])
    text[lattic_index+2] = "\t{}\t{}\t{}\n".format(lat[2][0], lat[2][1], lat[2][2])
    if save_scale_config is not None:
        with open(save_scale_config, 'w') as wf:
            wf.writelines(text)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', help='input', type=str, default='atom.config')
    parser.add_argument('-d', '--dim', help='"2 2 2" for 2x2x2 supercell', type=str, default='2 2 2')

    args = parser.parse_args()
    filename = args.inputfile
    dims = args.dim.strip().split()
    super_cell(dims=dims, filename=filename)