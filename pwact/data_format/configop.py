import os
from pwact.utils.constant import ELEMENTTABLE, DFT_STYLE, ELEMENTTABLE_2, CP2K, PWDATA
from pwact.utils.app_lib.cp2k import make_cp2k_xyz
from pwact.utils.file_operation import write_to_file
from pwdata.main import Config
from pwdata import perturb_structure, make_supercell, scale_cell
'''
description: 
    return atom type names and atom type numbers of config file
param {str} config_path
param {str} format
return {*}
author: wuxingxing
'''
def get_atom_type(config_path:str, format:str):
    image = Config.read(format=format, data_path=config_path, atom_names=None)
    atomic_number_list = []
    atomic_name_list = []
    for atom in image.atom_type:
        atomic_name_list.append(ELEMENTTABLE_2[atom])
        atomic_number_list.append(atom)
    return atomic_name_list, atomic_number_list

'''
description: 
    the input config could be vasp/pwmat/cp2k format
    for cp2k, the save_name will be 'coord.xyz'
return {*}
author: wuxingxing
'''
def save_config(config, input_format:str = None, wrap = False, direct = True, sort = True, \
        save_format:str=None, save_path:str=None, save_name:str=None, atom_names: list[str] = None):
    config = Config.read(format=input_format, data_path=config, atom_names=atom_names)
    if isinstance(config, list): # for lammps dump traj, config will be list
        config = config[0]
    if save_format == PWDATA.cp2k_scf:
        # make coord.xyz used by cp2k for every task 
        config = config._set_cartesian() if config.cartesian is False else config._set_cartesian()
        # potential = {"Si":"GTH-PBE"},
        # basis_set = {"Si":"DZVP-MOLOPT-SR-GTH-q4"}
        atom_types_image = []
        for atom in config.atom_types_image:
            atom_types_image.append(ELEMENTTABLE_2[atom])
        coord_xyz = make_cp2k_xyz(
            atom_types = atom_types_image,
            coord_list = config.position
        )
        write_to_file(os.path.join(save_path, save_name), coord_xyz, 'w')

        lattice = [element for sublist in config.lattice for element in sublist]
        lattice_line = ",".join(str(_) for _ in lattice)
        write_to_file(os.path.join(save_path, CP2K.cell_txt), lattice_line, 'w')

    else:
        config.to(output_path=save_path, 
                data_name    =save_name, 
                save_format  =save_format, 
                direct       =direct, 
                sort         =sort, 
                wrap         =wrap
                )
    
    return os.path.join(save_path, save_name)

def read_cp2k_xyz(config_file:str):
    with open(config_file, 'r') as rf:
            config_contents = rf.readlines()
    atom_names = []
    atom_type_name = []
    coord = []
    for line in config_contents:
        if line.strip() == 0:
            continue
        elements = line.split()
        atom_names.append(elements[0])
        if elements[0] not in atom_type_name:
            atom_type_name.append(elements[0])
        coord.appnd([float(elements[1]), float(elements[2]), float(elements[3])])
    return atom_type_name, atom_names, coord

def do_super_cell(config, input_format:str=None, supercell_matrix:list[int]=None, pbc:list[int]=[1, 1, 1], direct = True, sort = True, \
                    save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config.read(format=input_format, data_path=config, atom_names=None)
    # Make a supercell     
    supercell = make_supercell(config, supercell_matrix, pbc)
    # Write out the structure
    supercell.to(output_path = save_path,
                data_name = save_name,
                save_format = save_format,
                direct = direct,
                sort = sort)
    return os.path.join(save_path, save_name)

def do_scale(config, input_format:str=None, scale_factor:float=None, 
            direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config.read(format=input_format, data_path=config)
    scaled_struct = scale_cell(config, scale_factor)
    scaled_struct.to(output_path = save_path,
                    data_name = save_name,
                    save_format = save_format,
                    direct = direct,
                    sort = sort)
     
    return os.path.join(save_path, save_name)

def do_pertub(config, input_format:str=None, pert_num:int=None, cell_pert_fraction:float=None, atom_pert_distance:float=None, \
        direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config.read(format=input_format, data_path=config)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    perturbed_structs = perturb_structure(
            image_data = config,
            pert_num = pert_num,
            cell_pert_fraction = cell_pert_fraction,
            atom_pert_distance = atom_pert_distance)

    for tmp_perturbed_idx, tmp_pertubed_struct in enumerate(perturbed_structs):
        tmp_pertubed_struct.to(output_path = save_path,
                                data_name = "{}_{}".format(tmp_perturbed_idx, save_name),
                                save_format = save_format,
                                direct = direct,
                                sort = sort)

        print("pertub {} done!".format(os.path.join(save_path, "{}_{}".format(tmp_perturbed_idx, save_name))))

'''
description: 
    if merge is ture, save pwdata to datasets_path/data_name ...
    else:
        save pwdata to datasets_path/data_name/train or valid
return {*}
author: wuxingxing
'''
def extract_pwdata(data_list:list[str], 
                data_format:str="pwmat/movement", 
                datasets_path="PWdata", 
                train_valid_ratio:float=0.8, 
                data_shuffle:bool=True,
                merge_data:bool=False,
                interval:int=1
                ):
    # if data_format == DFT_STYLE.cp2k:
    #     raise Exception("not relized cp2k pwdata convert")

    data_name = None
    if merge_data:
        data_name = os.path.basename(datasets_path)
        if not os.path.isabs(datasets_path):
            # data_name = datasets_path
            datasets_path = os.path.dirname(os.path.join(os.getcwd(), datasets_path))
        else:
            datasets_path = os.path.dirname(datasets_path)
        image_data = None
        for data_path in data_list:
            if image_data is not None:
                tmp_config = Config(data_format, data_path)
                # if not isinstance(tmp_config, list):
                #     tmp_config = [tmp_config]
                image_data.append(tmp_config)
            else:
                image_data = Config(data_format, data_path)
                # if not isinstance(image_data, list):
                #     image_data = [image_data]
        image_data.to(
                    output_path=datasets_path,
                    save_format=PWDATA.pwmlff_npy,
                    data_name=data_name,
                    train_ratio = train_valid_ratio, 
                    train_data_path="train", 
                    valid_data_path="valid", 
                    random=data_shuffle,
                    seed = 2024, 
                    retain_raw = False
                    )
    else:
        for data_path in data_list:
            image_data = Config.read(data_format, data_path)
            image_data.to(
                output_path=datasets_path,
                save_format=PWDATA.pwmlff_npy,
                train_ratio = train_valid_ratio, 
                train_data_path="train", 
                valid_data_path="valid", 
                random=data_shuffle,
                seed = 2024, 
                retain_raw = False
                )
    
if __name__ == "__main__":
    in_config = "/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/collection/init_config_1/0.9_scale_pertub/0_pertub.config"
    save_path = "/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/collection/init_config_1/0.9_scale_pertub"
    image = Config.read(format="pwmat", data_path=in_config)
    save_config(image, wrap=False, direct=True, sort=True,\
        save_format="vasp", save_path=save_path, save_name="temp_poscar")
    # save_config(image, wrap=False, direct=True, sort=True,\
    #     save_format="vasp", save_path=save_path, save_name="temp_poscar")
    
    # do_super_cell(config=image,
    #         supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #         pbc=[1, 1, 1], 
    #         direct = True, 
    #         sort = True, \
    #         save_format="pwmat", save_path=save_path, save_name="temp_super_atom.config")
    
    # do_super_cell(config="/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/44_POSCAR",
    #     input_format="vasp",
    #     supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #     pbc=[1, 1, 1], 
    #     direct = True, 
    #     sort = True, \
    #     save_format="pwmat", save_path=save_path, save_name="temp_super_atom_from_poscar.config")
    
    
    # do_super_cell(config="/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/44_POSCAR",
    #     input_format="vasp",
    #     supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]], 
    #     pbc=[1, 1, 1], 
    #     direct = True, 
    #     sort = True, \
    #     save_format="vasp", save_path=save_path, save_name="temp_super_poscar")

    # do_scale(config=image,
    #         scale_factor=0.99, 
    #         direct=True,
    #         sort=True, 
    #         save_format="pwmat", save_path=save_path, save_name="temp_scale_atom.config")

    # do_scale(config=image,
    #         scale_factor=0.99, 
    #         direct=True,
    #         sort=True, 
    #         save_format="vasp", save_path=save_path, save_name="temp_scale_poscar")
    
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/temp_pwmat_pertub"
    # do_pertub(config=image, 
    #     pert_num=50, 
    #     cell_pert_fraction=0.01, 
    #     atom_pert_distance=0.04, 
    #     direct=True,
    #     sort=True, 
    #     save_format="pwmat", save_path=save_path, save_name="pertub.config")
    
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/sys_config/structures/temp_vasp_pertub"
    # do_pertub(config=image, 
    #     pert_num=50, 
    #     cell_pert_fraction=0.01, 
    #     atom_pert_distance=0.04, 
    #     direct=True,
    #     sort=True, 
    #     save_format="vasp", save_path=save_path, save_name="pertub.poscar")

    # convert trajs to config for model_deviation calculate
    # import glob
    # traj_dir = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/iter.0000/temp_run_iter_work/explore/md/md.000.sys.000/md.000.sys.000.t.000/traj"
    # save_dir = "/data/home/wuxingxing/datas/al_dir/si_5_pwmat/iter.0000/temp_run_iter_work/explore/md/md.000.sys.000/md.000.sys.000.t.000/traj2config"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # trajs = glob.glob(os.path.join(traj_dir, "*.lammpstrj"))
    # trajs = sorted(trajs)
    # for traj in trajs:
    #     save_name = "{}.config".format(os.path.basename(traj).split('.')[0])
    #     save_config(traj,  input_format="dump", wrap=False, direct=True, sort=True,\
    #     save_format="pwmat", save_path=save_dir, save_name=save_name)
    #     print("{} to {} done!".format(traj, save_name))