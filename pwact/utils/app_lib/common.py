from pwact.utils.constant import DFT_STYLE, VASP, PWMAT, CP2K, \
    ELEMENTTABLE_2, ELEMENTTABLE
from pwact.utils.file_operation import link_file, merge_files_to_one, write_to_file, copy_file, file_read_last_line, del_file
from pwact.utils.app_lib.pwmat import set_etot_input_by_file
import os
from pwact.data_format.configop import get_atom_type, read_cp2k_xyz
from pwact.utils.app_lib.cp2k import make_cp2k_input, make_cp2k_input_from_external
'''
description: 
    this script for pwmat, vasp, ... common operation
author: wuxingxing
'''

'''
description: 
    link the pseudo files to target dir
    for pwmat pseduo files, link each file to target dir
    for vasp pseduo files, merge files to one potcar file according to atom_order, then write to target dir
    for cp2k:
        copy the basis_set_file, potential_file_name to target dir
        
param {list} pseudo_list
param {str} target_dir
param {list} atom_order
param {str} dft_style
return {*}
author: wuxingxing
'''
def link_pseudo_by_atom(
        pseudo_list:list, 
        target_dir:str, 
        atom_order:list[str], 
        dft_style:str,
        basis_set_file:str=None,
        potential_file:str=None
    ):

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
        # assert len(pseudo_find) == len(atom_order), "the pwmat pseudo files {} not same as atom type '{}'".format(pseudo_find, atom_order)
        if basis_set_file is not None and potential_file is not None: # these 2 files for pwmat gaussian base
            link_file(basis_set_file, os.path.join(target_dir, os.path.basename(basis_set_file)))
            link_file(potential_file, os.path.join(target_dir, os.path.basename(potential_file)))

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
    
    elif dft_style == DFT_STYLE.cp2k:
        link_file(basis_set_file, os.path.join(target_dir, os.path.basename(basis_set_file)))
        link_file(potential_file, os.path.join(target_dir, os.path.basename(potential_file)))

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
    set input script, for pwmat is etot.input file, for vasp is INCAR file
return {*}
author: wuxingxing
'''
def set_input_script(            
    input_file:str=None,    # pwmat is etot.input, vasp is INCAR
    config:str=None, # the format is same as dft_style
    kspacing:float=None, 
    flag_symm:int=None, 
    dft_style:str=None,
    save_dir:str=None,
    pseudo_names:list[str]=None,
    is_scf = False, # if is_scf, the pwmat etot.input will set the 'out.mlmd = T'
    gaussian_base_param = None
    # basis_set_file_name=None,
    # potential_file_name=None,
    # basis_set_list=None,
    # potential_list=None
    # xc_functional=None,
    # potential=None,
    # basis_set=None
    ):
    if dft_style == DFT_STYLE.pwmat:
        target_file = os.path.join(save_dir, PWMAT.etot_input)
        is_skf_file = os.path.exists(os.path.join(save_dir, PWMAT.in_skf))
        script = set_etot_input_by_file(    #do pwmat etot.input content check
            etot_input_file=input_file, 
            atom_config=config, 
            kspacing=kspacing, 
            flag_symm=flag_symm, 
            pseudo_names=pseudo_names,
            is_scf = is_scf,
            is_skf_file = is_skf_file,
            gaussian_base_param=gaussian_base_param
            )
        write_to_file(target_file, script, "w")
    elif dft_style == DFT_STYLE.vasp:
        #do pwmat INCAR content check, not realized yet.
        target_file = os.path.join(save_dir, VASP.incar)
        copy_file(input_file, target_file)
    
    elif dft_style == DFT_STYLE.cp2k: # for cp2k is cp2k.inp file
        target_file = os.path.join(save_dir, CP2K.cp2k_inp)
        cell = file_read_last_line(os.path.join(save_dir, CP2K.cell_txt), type_name="float")
        del_file(os.path.join(save_dir, CP2K.cell_txt))
        # inp file, cell cood add to inp file
        # set kind_dict

        script = make_cp2k_input_from_external(
            cell=cell,
            coord_file = config,
            exinput_path=input_file,
            gaussian_base_param = gaussian_base_param
            )
        write_to_file(target_file, script, "w")

        # atom_type_name, atom_names, coord = read_cp2k_xyz(target_file) 
        # make_cp2k_input(
        #     cell=image.lattice,
        #     atom_names= atom_type_names, 
        #     atom_types=image.atom_type,
        #     coord_list=image.position,
        #     basis_set_file_name=basis_set_file_name,
        #     potential_file_name=potential_file_name,
        #     # xc_functional=potential_file_name,
        #     # potential=potential,
        #     # basis_set=basis_set
        #     )

    else:
        pass

def is_convergence(file_path, format):
    def _is_cvg_vasp(file_path:str):
        with open(file_path, 'r') as rf:
            outcar_contents = rf.readlines()
        nelm = None
        ediff = None
        for idx, ii in enumerate(outcar_contents):
            if 'NELM   =' in ii:
                nelm = int(ii.split()[2][:-1])
            if 'EDIFF = ' in ii:
                ediff = float(ii.split()[-1])
        
        with open(os.path.join(os.path.dirname(os.path.abspath(file_path)), "OSZICAR"), 'r') as rf:
            oszi_contents = rf.readlines()
        _split = oszi_contents[-2].split()
        real_nelm = int(_split[1])
        real_ediff1 = abs(float(_split[3]))
        real_ediff2 = abs(float(_split[4]))

        if real_nelm < nelm:
            return True
        elif real_ediff1 <= ediff and real_ediff2 <=ediff:
            return True
        else:
            False

    def _is_cvg_pwmat(file_path:str):
        with open(os.path.join(os.path.dirname(os.path.abspath(file_path)), "REPORT"), 'r') as rf:
            report_contents = rf.readlines()
        e_error   = None
        rho_error = None
        etot_idx = -1
        drho_idx = -1
        for idx, ii in enumerate(report_contents):
            if e_error is None and 'E_ERROR   =' in ii:
                e_error = abs(float(ii.split()[-1]))
            if rho_error is None and 'RHO_ERROR =' in ii:
                rho_error = abs(float(ii.split()[-1]))
            if 'E_tot(eV)            =' in ii:
                etot_idx = idx
            if 'dv_ave, drho_tot     =' in ii:
                drho_idx = idx
            if 'niter reached' in ii:
                break
            elif 'ending_scf_reason = tol' in ii:
                return True

        if e_error >= abs(float(report_contents[etot_idx].split()[-1])) or \
            rho_error >= abs(float(report_contents[drho_idx].split()[-1])):
            return True
        return False
    
    def _is_cvg_cp2k(file_path:str):
        with open(os.path.join(os.path.dirname(os.path.abspath(file_path)), "dft.log"), 'r') as rf:
            report_contents = rf.readlines()
        for idx, ii in enumerate(report_contents):
            if 'SCF run NOT converged' in ii:
                return False
        return True

    if format == DFT_STYLE.vasp:
        return _is_cvg_vasp(file_path)
    elif format == DFT_STYLE.pwmat:
        return _is_cvg_pwmat(file_path)
    elif format == DFT_STYLE.cp2k:
        return _is_cvg_cp2k(file_path)
    else: # for other format
        return True

def check_convergence(file_path:list[str], format:str):
    cvg_files = []
    uncvg_files = []
    cvg_infos = ""
    cvg_detail_infos=""
    for file in file_path:
        if is_convergence(file, format):
            cvg_files.append(file)
        else:
            uncvg_files.append(file)
    cvg_infos += "Number of converged files: {}, number of non-converged files: {}\n".format(len(cvg_files), len(uncvg_files))
    cvg_detail_infos += cvg_infos
    if len(uncvg_files) > 0:
        cvg_detail_infos += "List of non-converged files:\n{}".format("\n".join(uncvg_files))
    return cvg_files, uncvg_files, cvg_infos, cvg_detail_infos
    