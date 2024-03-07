'''
description: 
    this a script tool for cp2k input file constructing.
    The source data is if from github:https://github.com/deepmodeling/dpgen/blob/master/dpgen/generator/lib/cp2k.py

return {*}
author: wuxingxing
'''

import numpy as np
default_config = {
    "GLOBAL": {"PROJECT": "AL_PWMLFF"},
    "FORCE_EVAL": {
        "METHOD": "QS",
        "STRESS_TENSOR": "ANALYTICAL",
        "DFT": {
            "BASIS_SET_FILE_NAME": "BASIS_MOLOPT",
            "POTENTIAL_FILE_NAME": "GTH_POTENTIALS",
            "CHARGE": 0,
            "UKS": "F",
            "MULTIPLICITY": 1,
            "MGRID": {"CUTOFF": 400, "REL_CUTOFF": 50, "NGRIDS": 4},
            "QS": {"EPS_DEFAULT": "1.0E-12"},
            "SCF": {"SCF_GUESS": "ATOMIC", "EPS_SCF": "1.0E-6", "MAX_SCF": 50},
            "XC": {"XC_FUNCTIONAL": {"_": "PBE"}},
        },
        "SUBSYS": {
            "CELL": {"A": "10 .0 .0", "B": ".0 10 .0", "C": ".0 .0 10"},
            "COORD": {"@include": "coord.xyz"},
            "KIND": {
                "_": ["H", "C", "N"],
                "POTENTIAL": ["GTH-PBE-q1", "GTH-PBE-q4", "GTH-PBE-q5"],
                "BASIS_SET": ["DZVP-MOLOPT-GTH", "DZVP-MOLOPT-GTH", "DZVP-MOLOPT-GTH"],
            },
        },
        "PRINT": {"FORCES": {"_": "ON"}, "STRESS_TENSOR": {"_": "ON"}},
    },
}


def update_dict(old_d, update_d):
    """A method to recursive update dict
    :old_d: old dictionary
    :update_d: some update value written in dictionary form.
    """
    import collections.abc

    for k, v in update_d.items():
        if (
            k in old_d
            and isinstance(old_d[k], dict)
            and isinstance(update_d[k], collections.abc.Mapping)
        ):
            update_dict(old_d[k], update_d[k])
        else:
            old_d[k] = update_d[k]


def iterdict(d, out_list, flag=None):
    """:doc: a recursive expansion of dictionary into cp2k input
    :k: current key
    :v: current value
    :d: current dictionary under expansion
    :flag: used to record dictionary state. if flag is None,
    it means we are in top level dict. flag is a string.
    """
    for k, v in d.items():
        k = str(k)  # cast key into string
        # if value is dictionary
        if isinstance(v, dict):
            # flag == None, it is now in top level section of cp2k
            if flag is None:
                out_list.append("&" + k)
                out_list.append("&END " + k)
                iterdict(v, out_list, k)
            # flag is not None, now it has name of section
            else:
                index = out_list.index("&END " + flag)
                out_list.insert(index, "&" + k)
                out_list.insert(index + 1, "&END " + k)
                iterdict(v, out_list, k)
        elif isinstance(v, list):
            #            print("we have encountered the repeat section!")
            index = out_list.index("&" + flag)
            # delete the current constructed repeat section
            del out_list[index : index + 2]
            # do a loop over key and corresponding list
            k_tmp_list = []
            v_list_tmp_list = []
            for k_tmp, v_tmp in d.items():
                k_tmp_list.append(str(k_tmp))
                v_list_tmp_list.append(v_tmp)
            for repeat_keyword in zip(*v_list_tmp_list):
                out_list.insert(index, "&" + flag)
                out_list.insert(index + 1, "&END " + flag)
                for idx, k_tmp in enumerate(k_tmp_list):
                    if k_tmp == "_":
                        out_list[index] = "&" + flag + " " + repeat_keyword[idx]
                    else:
                        out_list.insert(index + 1, k_tmp + " " + repeat_keyword[idx])

            break

        else:
            v = str(v)
            if flag is None:
                out_list.append(k + " " + v)
                print(k, ":", v)
            else:
                if k == "_":
                    index = out_list.index("&" + flag)
                    out_list[index] = "&" + flag + " " + v

                else:
                    index = out_list.index("&END " + flag)
                    out_list.insert(index, k + " " + v)


def make_cp2k_input(
        cell:list[float],
        atom_names:list[str], 
        basis_set_file_name:str,
        potential_file_name:str,
        xc_functional:str,
        potential:dict,
        basis_set:dict,
        coord_content:str = None # not used
        ):
    # covert cell to cell string
    cell = np.reshape(cell, [3, 3])
    cell_a = np.array2string(cell[0, :])
    cell_a = cell_a[1:-1]
    cell_b = np.array2string(cell[1, :])
    cell_b = cell_b[1:-1]
    cell_c = np.array2string(cell[2, :])
    cell_c = cell_c[1:-1]
    # get update from cell
    in_potentail =[]
    in_basis_set = []
    for atom in atom_names:
        in_potentail.append(potential[atom])
        in_basis_set.append(basis_set[atom])

    cell_config = {
        "FORCE_EVAL": {
            "SUBSYS": {
                        "BASIS_SET_FILE_NAME": basis_set_file_name,
                        "POTENTIAL_FILE_NAME": potential_file_name,
                        "CELL": {"A": cell_a, "B": cell_b, "C": cell_c},
                        # "COORD": coord_content, 
                        "COORD": {"@include": "coord.xyz"}, 
                        "KIND": {
                        "_": atom_names,
                        "POTENTIAL": in_potentail,
                        "BASIS_SET": in_basis_set
                        },
                        "XC": {"XC_FUNCTIONAL": {"_": xc_functional}}
            }
        }
    }
    update_dict(default_config, cell_config)
    # output list
    input_str = []
    iterdict(default_config, input_str)
    string = "\n".join(input_str)
    return string

def make_cp2k_xyz(atom_types:list[int], coord_list:list[float]):
    # get structral information
    # write coordinate to xyz file used by cp2k input
    atom_list = np.array(atom_types)
    x = "\n"
    for kind, coord in zip(atom_list, coord_list):
        x += str(kind) + " " + str(coord[:])[1:-1] + "\n"
    return x

'''
description: 
    delete cell and coord
    1.for coord, set it :"COORD": {"@include": "coord.xyz"}
    2.for cell, set it with input
    3.set 'PRINT_LEVEL' to 'medium'
    attation, the cell block and coord block has no order
param {*} sys_data
param {*} exinput_path
return {*}
author: wuxingxing
'''
def make_cp2k_input_from_external(cell, coord_file_name, exinput_path):
   
    # insert the cell information
    # covert cell to cell string
    cell = np.reshape(cell, [3, 3])
    # read the input content as string
    with open(exinput_path) as f:
        exinput = f.readlines()
    # replace the cell string
    start_cell = 0
    end_cell = 0
    start_subsys = 0
    end_subsys = 0
    start_coord = 0
    end_coord = 0
    start_global = 0
    end_global = 0
    print_level_line = -1

    for line_idx, line in enumerate(exinput):
        line = line.upper()
        if "&GLOBAL" in line:
            start_global = line_idx
        if "&END GLOBAL" in line:
            end_global = line_idx
        if "PRINT_LEVEL" in line:
            print_level_line = line_idx
        if "&SUBSYS" in line:
            start_subsys = line_idx
        if "&END SUBSYS" in line:
            end_subsys = line_idx
        if "&CELL" in line:
            start_cell = line_idx
        if "&END CELL" in line:
            end_cell = line_idx
        if "&COORD" in line:
            start_coord = line_idx
        if "&END COORD" in line:
            end_coord = line_idx
    if start_global == end_global:
        raise Exception("ERROR! the input cp2k inp file does not have 'GLOBAL' block! Please check the file {}\n".format(exinput_path))
    temp_exinput = exinput[:start_subsys+1]
    # add coord
    temp_exinput.append("    &COORD\n")
    temp_exinput.append("        @include {}\n".format(coord_file_name))
    # temp_exinput.append("        COORD_FILE_FORMAT XYZ\n")
    temp_exinput.append("    &END COORD\n")
    #add cell
    temp_exinput.append("    &CELL\n")
    temp_exinput.append("        A     {}    {}     {}\n".format(cell[0, 0], cell[0, 1], cell[0, 2]))
    temp_exinput.append("        B     {}    {}     {}\n".format(cell[1, 0], cell[1, 1], cell[1, 2]))
    temp_exinput.append("        C     {}    {}     {}\n".format(cell[2, 0], cell[2, 1], cell[2, 2]))
    # temp_exinput.append("        PERIODIC XYZ\n")
    temp_exinput.append("    &END CELL\n")

    del_content_index = []
    if start_cell != end_cell:
        del_content_index.extend(list(range(start_cell, end_cell+1)))
    if start_coord != end_coord:
        del_content_index.extend(list(range(start_coord, end_coord+1)))
    del_content_index = sorted(del_content_index)
    for index in range(start_subsys+1, end_subsys):
        if index not in del_content_index:
            temp_exinput.append(exinput[index])
    temp_exinput.extend(exinput[end_subsys:])

    #reset or add print_level medium
    if print_level_line == -1:
        temp_exinput.insert(end_global-1, "    PRINT_LEVEL medium\n")
    else:
        temp_exinput[print_level_line] = "    PRINT_LEVEL medium\n"
    return "".join(temp_exinput)

# if __name__=="__main__":
    # import dpdata
    # poscar = "/data/home/wuxingxing/datas/al_dir/si_4_vasp/init_bulk/collection/init_config_0/0.9_scale.poscar"
    # sys_data = dpdata.System(poscar).data
    
    # from pwdata.main import Configs
    # from pwdata.calculators.const import ELEMENTTABLE_2
    # image = Configs.read(format="pwmat", data_path="/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/atom.config")
    # image = image._set_cartesian() if image.cartesian is False else image._set_cartesian()
    # potential = {"Si":"GTH-PBE"}
    # basis_set = {"Si":"DZVP-MOLOPT-SR-GTH-q4"}
    # atom_types_image = []
    # for atom in image.atom_types_image:
    #     atom_types_image.append(ELEMENTTABLE_2[atom])
    # coord_xyz = make_cp2k_xyz(
    #     atom_types = atom_types_image,
    #     coord_list = image.position
    # )
    # with open("/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/coord.xyz", "w") as fp:
    #     fp.write(coord_xyz)
    
    # make_cp2k_input(
    #     cell = image.lattice,
    #     atom_names=["Si"],
    #     basis_set_file_name="BASIS_SET_FILE",
    #     potential_file_name="POTENTIAL_FILE",
    #     xc_functional="PBE",
    #     potential=potential,
    #     basis_set=basis_set,
    #     coord_content=coord_xyz
    # )