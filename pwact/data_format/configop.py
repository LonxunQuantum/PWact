import os
from pwact.utils.constant import ELEMENTTABLE, DFT_STYLE, ELEMENTTABLE_2, CP2K, PWDATA
from pwact.utils.app_lib.cp2k import make_cp2k_xyz
from pwact.utils.file_operation import write_to_file
from pwdata.config import Config
from pwdata import perturb_structure, make_supercell, scale_cell
'''
description: 
    return atom type names and atom type numbers of config file
param {str} config_path
param {str} format
return {*}
author: wuxingxing
'''
def get_atom_type(config_path, format:str=None):
    if isinstance(config_path, str):
        image = Config(format=format, data_path=config_path, atom_names=None).images[0]
    else:
        image = config_path
    atomic_number_list = []
    atomic_name_list = []
    for atom in image.atom_type:
        atomic_name_list.append(ELEMENTTABLE_2[atom])
        atomic_number_list.append(atom)
    return atomic_name_list, atomic_number_list

def load_config(config, format, atom_names=None):
    config = Config(format=format, data_path=config, atom_names=atom_names)
    return config

'''
description: 
    the input config could be vasp/pwmat/cp2k format
    for cp2k, the save_name will be 'coord.xyz'
return {*}
author: wuxingxing
'''
def save_config(config, input_format:str = None, wrap = False, direct = True, sort = True, \
        save_format:str=None, save_path:str=None, save_name:str=None, atom_names: list[str] = None):
    if isinstance(config, str):
        config = Config(format=input_format, data_path=config, atom_names=atom_names).images[0]
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
        config.to(data_path  =save_path, 
                data_name    =save_name, 
                format       =save_format, 
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

def do_super_cell(config_file, input_format:str=None, supercell_matrix:list[int]=None, pbc:list[int]=[1, 1, 1], direct = True, sort = True, \
                    save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config(format=input_format, data_path=config_file, atom_names=None)
    # Make a supercell     
    supercell = make_supercell(config, supercell_matrix, pbc)
    # Write out the structure
    supercell.to(data_path = save_path,
                data_name  = save_name,
                format     = save_format,
                direct = direct,
                sort = sort)
    return os.path.join(save_path, save_name)

def do_scale(config, input_format:str=None, scale_factor:float=None, 
            direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config(format=input_format, data_path=config)
    scaled_struct = scale_cell(config, scale_factor)
    scaled_struct.to(data_path = save_path,
                    data_name  = save_name,
                    format     = save_format,
                    direct = direct,
                    sort = sort)
     
    return os.path.join(save_path, save_name)

def do_pertub(config, input_format:str=None, pert_num:int=None, cell_pert_fraction:float=None, atom_pert_distance:float=None, \
        direct:bool=True, sort:bool=True, save_format:str=None, save_path:str=None, save_name:str=None):
    config = Config(format=input_format, data_path=config)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    perturbed_structs = perturb_structure(
            image_data = config,
            pert_num = pert_num,
            cell_pert_fraction = cell_pert_fraction,
            atom_pert_distance = atom_pert_distance)

    for tmp_perturbed_idx, tmp_pertubed_struct in enumerate(perturbed_structs):
        tmp_pertubed_struct.to(data_path  = save_path,
                                data_name = "{}_{}".format(tmp_perturbed_idx, save_name),
                                format    = save_format,
                                direct = direct,
                                sort = sort)

        print("pertub {} done!".format(os.path.join(save_path, "{}_{}".format(tmp_perturbed_idx, save_name))))

'''
description: 
    save the inputfiles to pwmlff/npy format data
return {*}
author: wuxingxing
'''
def extract_pwdata(input_data_list:list[str], 
                intput_data_format:str="pwmat/movement", 
                save_data_path:str="./",
                save_data_name="PWdata", 
                save_data_format="extxyz",
                data_shuffle:bool=False,
                interval:int=1
                ):
    # if data_format == DFT_STYLE.cp2k:
    #     raise Exception("not relized cp2k pwdata convert")

    if not os.path.isabs(save_data_path):
        # data_name = datasets_path
        save_data_path = os.path.join(os.getcwd(), save_data_path)
    image_data = None
    for dir in input_data_list:
        if image_data is not None:
            tmp_config = Config(format=intput_data_format, data_path=dir)
            # if not isinstance(tmp_config, list):
            #     tmp_config = [tmp_config]
            image_data.images.extend(tmp_config.images)
        else:
            image_data = Config(format=intput_data_format, data_path=dir)
            
            if not isinstance(image_data.images, list):
                image_data.images = [image_data.images]
        
            # if not isinstance(image_data, list):
            #     image_data = [image_data]
    if interval > 1:
        tmp = []
        for i in range(0, len(image_data.images)):
            if i % interval == 0:
                tmp.append(image_data.images[i])
        image_data.images = tmp

    image_data.to(
                data_path  =save_data_path,
                data_name  =save_data_name,
                format     =save_data_format,
                random=data_shuffle
                )
    
if __name__ == "__main__":
    # in_config = "/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/collection/init_config_1/0.9_scale_pertub/0_pertub.config"
    # save_path = "/data/home/wuxingxing/datas/al_dir/si_exp/init_bulk/collection/init_config_1/0.9_scale_pertub"
    # image = Config.read(format="pwmat", data_path=in_config)
    # save_config(image, wrap=False, direct=True, sort=True,\
    #     save_format="vasp", save_path=save_path, save_name="temp_poscar")
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
    import glob
    data_dir1 = glob.glob(os.path.join("/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/temp_init_bulk_work/scf/init_config_*")) #
    data_dir2 = glob.glob(os.path.join("/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2_600k/temp_init_bulk_work/scf/init_config_*")) 
    data_dir1.extend(data_dir2)
    select_list = ["0-scf", "200-scf", "400-scf", "600-scf", "800-scf", "1000-scf"]
    data_list = []
    for dir in data_dir1:
        _outcars = glob.glob(os.path.join(dir, "*/*/*/OUTCAR"))
        for outcar in _outcars:
            if os.path.basename(os.path.dirname(outcar)) in select_list:
                data_list.append(outcar)

    datasets_path = "/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_data_200"
    extract_pwdata(input_data_list=data_list, 
            intput_data_format="vasp/outcar", 
            save_data_path=datasets_path
            )