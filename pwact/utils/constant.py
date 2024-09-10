'''
description: 
    Active learning of top-level directory structure
return {*}
author: wuxingxing
'''
class AL_WORK:
    init_bulk = "init_bulk"
    init_surface = "init_surface"
    run_iter = "run"

class AL_STRUCTURE:
    train = "train"
    explore = "explore"
    labeling = "label"
    pertub = "pertub"
    aimd = "aimd"
    collection = "collection"
    init_data = "init_data_npy"

'''
description: 
    slurm job log file name
return {*}
author: wuxingxing
'''
class SLURM_OUT:
    dft_out = "dft.log"
    train_out = "train.log"
    md_out = "md.log"
    kpu_out = "kpu.log"

class SLURM_JOB_TYPE:
    cp2k_relax = "cp2k/relax"
    cp2k_scf = "cp2k/scf"
    cp2k_aimd = "cp2k/aimd"
    pwmat_relax = "pwmat/relax"
    pwmat_scf = "pwmat/scf"
    pwmat_aimd = "pwmat/aimd"
    vasp_relax = "vasp/relax"
    vasp_scf = "vasp/scf"
    vasp_aimd = "vasp/aimd"
    lammps = "lammps"

'''
description: 
    training commands
return {*}
author: wuxingxing
'''
class MODEL_CMD:
    train = "train"
    gen_feat = "gen_feat"
    test = "test"
    script = "script"
    compress = "compress"
    kpu = "kpu"
    pwdata =  "pwdata"

'''
description: 
    model training parametors
return {*}
author: wuxingxing
'''
class TRAIN_INPUT_PARAM:
    save_p_matrix = "save_p_matrix" # for kpu
    raw_files = "raw_files"
    datasets_path = "datasets_path"
    test_mvm_files = "test_movement_file"
    reserve_feature = "reserve_feature" #False
    reserve_work_dir = "reserve_work_dir" #False
    valid_shuffle = "valid_shuffle" #True
    train_valid_ratio = "train_valid_ratio" #0.8
    sample_nums ="sample_nums" # for kpu or test nums
    seed = "seed" #2023
    recover_train = "recover_train" #true
    type_embedding = "type_embedding"
    model_type = "model_type"
    atom_type = "atom_type"
    model_load_file = "model_load_file"
    test_dir_name = "test_dir_name"
    work_dir = "work_dir"
    optimizer = "optimizer"
    reset_epoch = "reset_epoch"
    #epoch

'''
description: 
    lammps md commands
return {*}
author: wuxingxing
'''
class LAMMPS_CMD:
    lmp_mpi = "lmp_mpi"
    lmp_mpi_gpu = "lmp_mpi_gpu"

'''
description: 
    force filed types
return {*}
author: wuxingxing
'''
class FORCEFILED:
    fortran_lmps = 1 # use cpu dp model
    libtorch_lmps = 2 # default, use jit model or nep cpu/gpu model
    main_md = 3 #

'''
description: 
    pwdata format
return {*}
author: wuxingxing
'''
class PWDATA:
    pwmat_config = "pwmat/config"
    vasp_poscar = "vasp/poscar"
    lammps_dump = "lammps/dump"
    lammps_lmp = "lammps/lmp"
    pwmat_movement = "pwmat/movement"
    vasp_outcar = "vasp/outcar"
    extxyz = "extxyz"
    vasp_xml = "vasp/xml"
    cp2k_md = 'cp2k/md'
    cp2k_scf = 'cp2k/scf'
    pwmlff_npy = "pwmlff/npy"

'''
description: 
    DFT apps command
return {*}
author: wuxingxing
'''
class DFT_STYLE:
    vasp = "vasp"
    pwmat = "pwmat"
    cp2k = "cp2k"
    lammps = "lammps"

    '''
    description: 
        is_cp2k_coord: if ture, for cp2k, return 'cp2k/scf', it is used to extract config from cp2k log file
    return {*}
    author: wuxingxing
    '''
    @staticmethod
    def get_pwdata_format(dft_style:str, is_cp2k_coord:bool=False):
        if dft_style.lower() == DFT_STYLE.pwmat.lower() : # atom.config
            return PWDATA.pwmat_config
        if dft_style.lower() == DFT_STYLE.vasp.lower(): 
            return PWDATA.vasp_poscar
        if dft_style.lower() == DFT_STYLE.cp2k.lower(): 
            if is_cp2k_coord:
                return PWDATA.cp2k_scf
            else:
                return PWDATA.vasp_poscar

    @staticmethod
    def get_normal_config(dft_style:str): # the input config file name of pwmat vasp and cp2k
        if dft_style == DFT_STYLE.pwmat: # atom.config
            return PWMAT.atom_config
        elif dft_style == DFT_STYLE.vasp: # poscar
            return VASP.poscar
        elif dft_style == DFT_STYLE.cp2k: # coord.xyz, for cp2k, the position of atoms writed in coord.xyz file, the cell is in inp file
            return CP2K.coord_xyz
    
    @staticmethod
    def get_postfix(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return ".config"
        elif dft_style == DFT_STYLE.vasp:
            return ".poscar"
        elif dft_style == DFT_STYLE.cp2k:
            return ".poscar"

    @staticmethod
    def get_format_by_postfix(file_name:str):
        if "config" in file_name.lower()\
            or "pwmat" in file_name.lower():
            return PWDATA.pwmat_config
        elif "movement" in file_name.lower():
            return PWDATA.pwmat_movement
        elif "out.mlmd" in file_name.lower():
            return PWDATA.pwmat_movement
            
        elif "poscar" in file_name.lower() or\
            "contcor" in file_name.lower() or\
                "vasp" in file_name.lower():
            return PWDATA.vasp_poscar

        elif "outcar" in file_name.lower():
            return PWDATA.vasp_outcar

        elif "inp" in file_name.lower() or\
            "xyz" in file_name.lower() or\
                "cp2k" in file_name.lower():
            return PWDATA.cp2k_scf

        elif CP2K.final_config.lower() in file_name.lower():
            return PWDATA.cp2k_scf
        
    '''
    description: 
        for pwmat is final.config
        for vasp is  CONTCAR
    return {*}
    author: wuxingxing
    '''    
    @staticmethod
    def get_relaxed_original_name(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return PWMAT.final_config
        elif dft_style == DFT_STYLE.vasp:
            return VASP.final_config
        elif dft_style == DFT_STYLE.cp2k:
            return CP2K.final_config


    @staticmethod
    def get_scf_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return PWMAT.out_mlmd
        elif dft_style == DFT_STYLE.vasp:
            return VASP.outcar
        elif dft_style == DFT_STYLE.cp2k:
            return CP2K.final_config

    '''
    description: 
        scf files need reserved
            for vasp is outcar poscar and incar file
            for pwmat is atom.config etot.input report and out.mlmd file
                the scf of pwmat will produce a file named final.config, does not reserve
    param {str} dft_style
    return {*}
    author: wuxingxing
    '''
    @staticmethod
    def get_scf_reserve_list(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            scf_list = PWMAT.scf_reserve_list
        elif dft_style == DFT_STYLE.vasp:
            scf_list = VASP.scf_reserve_list
        elif dft_style == DFT_STYLE.cp2k:
            scf_list = CP2K.scf_reserve_list

        scf_list = [_.lower() for _ in scf_list]
        return scf_list
    
    '''
    description: 
        the files in scf does not need reserve
    return {*}
    author: wuxingxing
    '''    
    @staticmethod
    def get_scf_del_list():
        del_list = ["final.config"]
        return del_list
    
    @staticmethod
    def get_aimd_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return PWMAT.MOVEMENT
        elif dft_style == DFT_STYLE.vasp:
            return VASP.outcar
        elif dft_style == DFT_STYLE.cp2k:# for cp2k, convert the output content to poscar format
            return CP2K.final_config

    @staticmethod
    def get_aimd_config_format(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return PWDATA.pwmat_movement
        elif dft_style == DFT_STYLE.vasp:
            return PWDATA.vasp_outcar
        elif dft_style == DFT_STYLE.cp2k:# for cp2k, convert the output content to poscar format
            return PWDATA.cp2k_md

    @staticmethod
    def get_pertub_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return "pertub.config"
        elif dft_style == DFT_STYLE.vasp:
            return "pertub.poscar"
        elif dft_style == DFT_STYLE.cp2k:# for cp2k, convert the output content to poscar format
            return "pertub.poscar"

    @staticmethod
    def get_super_cell_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return "super_cell.config"
        elif dft_style == DFT_STYLE.vasp:
            return "super_cell.poscar"
        elif dft_style == DFT_STYLE.cp2k:# for cp2k, convert the output content to poscar format
            return "super_cell.poscar"

    
    @staticmethod
    def get_scale_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return "scale.config"
        elif dft_style == DFT_STYLE.vasp:
            return "scale.poscar"
        elif dft_style == DFT_STYLE.cp2k:# for cp2k, convert the output content to poscar format
            return "scale.poscar"

    @staticmethod
    def get_relaxed_config(dft_style:str):
        if dft_style == DFT_STYLE.pwmat:
            return "relaxed.config"
        elif dft_style == DFT_STYLE.vasp:
            return "relaxed.poscar"
        elif dft_style == DFT_STYLE.cp2k: # for cp2k, convert the output content to poscar format
            return "relaxed.poscar"
    
class DFT_TYPE:
    relax = "relax"
    aimd = "aimd"
    scf = "scf"

'''
description: 
    temp files name
return {*}
author: wuxingxing
'''
class TEMP_STRUCTURE:
    tmp_init_bulk_dir = "temp_init_bulk_work"
    tmp_run_iter_dir = "temp_run_iter_work"
    tmp_prefix = "temp"
    
class INIT_BULK:
    relax = "relax"
    relax_job ="relax.job"
    relax_tag = "tag.relax.success"
    relax_tag_failed = "tag.relax.failed"
    
    init_config = "init_config"
    init = "init"
    super_cell_scale = "super_cell_scale"
    tag_super_cell = "tag.success.super_cell_scale"
    super_cell = "super_cell"
    scale = "scale"
    
    pertub = "pertub"
    tag_pertub = "tag.success.pertub"
    
    aimd = "aimd"
    aimd_job = "aimd.job"
    aimd_tag = "tag.aimd.success"
    aimd_tag_failed ="tag.aimd.failed"
    
    scf = "scf"
    scf_job = "scf.job"
    scf_tag = "tag.scf.success"
    scf_tag_failed ="tag.scf.failed"

    collection = "collection"
    npy_format_save_dir = "PWdata"
    npy_format_name = "pwdata"

class MODEL_TYPE:
    dp = "DP"
    nep = "NEP"

class TRAIN_FILE_STRUCTUR:
    work_dir = "work_dir"
    feature_dir = "feature"
    feature_json = "feature.json"
    feature_job = "feature.job"
    feature_tag = "tag.feature.success"
    feature_tag_failed = "tag.feature.failed"
    train_json = "train.json"
    train_job = "train.job"
    train_tag = "tag.train.success"
    train_tag_failed = "tag.train.failed"
    
    kpu_json = "kpu.json"
    kpu_tag = "tag.kpu.success"
    kpu_tag_failed = "tag.kpu.failed"
    kpu_job = "kpu.job"
    kpu = "kpu"
    kpu_file = "kpu_info.csv"
    base_kpu = "base_kpu"
    
    movement = "MOVEMENT"
    model_record = "model_record"
    # dp model
    dp_model_name ="dp_model.ckpt"
    compree_dp_name = "cmp_dp_model.ckpt"
    # cmp_tracing_dp_name = "torch_script_module.pt"
    script_dp_name = "torch_script_module.pt"
    script_dp_name_cpu = "jit_dp_cpu.pt"
    script_dp_name_gpu = "jit_dp_gpu.pt"
    fortran_dp = "forcefield"
    fortran_dp_name = "forcefield.ff"

    # nep model
    nep_model_name ="nep_model.ckpt"
    nep_model_lmps = "nep_to_lmps.txt"

class EXPLORE_FILE_STRUCTURE:
    kpu= "kpu"
    md = "md"
    select = "select"
    md_tag = "tag.md.success"
    md_tag_faild = "tag.md.error"
    md_job = "md.job"
    # selected image info file names
    candidate = "candidate.csv"
    # candidate_random = "candidate_random.csv"
    candidate_delete = "candidate_delete.csv"
    failed = "fail.csv"
    accurate = "accurate.csv"
    select_summary = "select_summary.txt"

    traj ="traj"
    
    # for committee and kpu method 
    model_devi = "model_devi.out"
    kpu_model_devi = "kpu_model_devi.out"
    devi_columns = ["devi_force", "config_index", "file_path"]

    iter_select_file = "iter_select.txt"

    @staticmethod
    def get_devi_name(data_type:str):
        if data_type == UNCERTAINTY.committee:
            devi_name = EXPLORE_FILE_STRUCTURE.model_devi
        elif data_type == UNCERTAINTY.kpu:
            devi_name = EXPLORE_FILE_STRUCTURE.kpu_model_devi
        else:
            raise Exception("get_devi_name error, the data_type {} not relized".format(data_type))
        return devi_name


class LABEL_FILE_STRUCTURE:
    scf = "scf"
    result = "result"
    scf_tag = "tag.scf.success"
    scf_tag_failed = "tag.scf.failed"
    scf_job = "scf.job"

class LAMMPS:
    input_lammps="in.lammps"
    poscar = "POSCAR"
    lammps_sys_config = "lmp.config"
    traj_postfix = ".lammpstrj"
    log_lammps = "log.lammps"
    traj_format = "dump"
    lmp_format = "lmp"
    atom_type_file = "atom_type.txt"

class ENSEMBLE:
    npt_tri = "npt_tri",
    nvt = "nvt"
    npt = "npt"
    nve = "nve"

class PWMAT:
    in_skf = "in_skf"
    pwmat_out = "PWMAT.out"
    config_postfix = ".config"
    atom_config = "atom.config"
    etot_input = "etot.input"
    out_mlmd = "OUT.MLMD"
    mvm = "mvm"
    relax="RELAX"
    scf = "SCF"
    md = "MD"
    MOVEMENT="MOVEMENT"
    MOVEMENT_low = "movement"
    kspacing_default = 0.5
    scf_reserve_list = ["REPORT", "etot.input","OUT.MLMD", ".config", "MOVEMENT"]
    final_config = "final.config"#relaxed result

class VASP:
    potcar = "POTCAR"
    incar = "INCAR"
    poscar = "POSCAR"
    final_config = "CONTCAR"#relaxed result
    outcar = "OUTCAR"
    scf_reserve_list = ["OUTCAR","POSCAR", "INCAR"]

class CP2K:
    cp2k_inp  = "cp2k.inp"
    coord_xyz = "coord.xyz"
    cell_txt = "cell.txt"
    traj_xyz = "traj.xyz"
    final_config = "dft.log" # relaxed result , the cp2k output should be extraced from log
    scf_reserve_list = ["cp2k.inp", "dft.log", "coord.xyz"] # scf reserve file list

class UNCERTAINTY:
    kpu = "KPU"
    committee="COMMITTEE"

ELEMENTTABLE={'H': 1,
    'He': 2,  'Li': 3,  'Be': 4,  'B': 5,   'C': 6,   'N': 7,   'O': 8,   'F': 9,   'Ne': 10,  'Na': 11,
    'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,  'S': 16,  'Cl': 17, 'Ar': 18, 'K': 19,  'Ca': 20,  'Sc': 21,
    'Ti': 22, 'V': 23,  'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,  'Ga': 31,
    'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39,  'Zr': 40,  'Nb': 41, 
    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,  'Sb': 51, 
    'Te': 52, 'I': 53,  'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,  'Pm': 61,
    'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,  'Lu': 71, 
    'Hf': 72, 'Ta': 73, 'W': 74,  'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,  'Tl': 81, 
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,  'Pa': 91, 
    'U': 92,  'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
    'No': 102,'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,'Bh': 107,'Hs': 108,'Mt': 109,'Ds': 110,'Rg': 111,
    'Uub': 112
    }

ELEMENTTABLE_2 = {1: 'H', 
    2: 'He',     3: 'Li',   4: 'Be',   5: 'B',    6: 'C',    7: 'N',   8: 'O',     9: 'F',   10: 'Ne',  11: 'Na', 
    12: 'Mg',   13: 'Al',  14: 'Si',  15: 'P',   16: 'S',   17: 'Cl',  18: 'Ar',  19: 'K',   20: 'Ca',  21: 'Sc', 
    22: 'Ti',   23: 'V',   24: 'Cr',  25: 'Mn',  26: 'Fe',  27: 'Co',  28: 'Ni',  29: 'Cu',  30: 'Zn',  31: 'Ga', 
    32: 'Ge',   33: 'As',  34: 'Se',  35: 'Br',  36: 'Kr',  37: 'Rb',  38: 'Sr',  39: 'Y',   40: 'Zr',  41: 'Nb', 
    42: 'Mo',   43: 'Tc',  44: 'Ru',  45: 'Rh',  46: 'Pd',  47: 'Ag',  48: 'Cd',  49: 'In',  50: 'Sn',  51: 'Sb', 
    52: 'Te',   53: 'I',   54: 'Xe',  55: 'Cs',  56: 'Ba',  57: 'La',  58: 'Ce',  59: 'Pr',  60: 'Nd',  61: 'Pm', 
    62: 'Sm',   63: 'Eu',  64: 'Gd',  65: 'Tb',  66: 'Dy',  67: 'Ho',  68: 'Er',  69: 'Tm',  70: 'Yb',  71: 'Lu', 
    72: 'Hf',   73: 'Ta',  74:  'W',  75: 'Re',  76: 'Os',  77: 'Ir',  78: 'Pt',  79: 'Au',  80: 'Hg',  81: 'Tl', 
    82: 'Pb',   83: 'Bi',  84: 'Po',  85: 'At',  86: 'Rn',  87: 'Fr',  88: 'Ra',  89: 'Ac',  90: 'Th',  91: 'Pa', 
    92: 'U',    93: 'Np',  94: 'Pu',  95: 'Am',  96: 'Cm',  97: 'Bk',  98: 'Cf',  99: 'Es', 100: 'Fm', 101: 'Md', 
    102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 
    112: 'Uub'
    }

ELEMENTMASSTABLE={  1:1.007,2:4.002,3:6.941,4:9.012,5:10.811,6:12.011,
                            7:14.007,8:15.999,9:18.998,10:20.18,11:22.99,12:24.305,
                            13:26.982,14:28.086,15:30.974,16:32.065,17:35.453,
                            18:39.948,19:39.098,20:40.078,21:44.956,22:47.867,
                            23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
                            28:58.693,29:63.546,30:65.38,31:69.723,32:72.64,33:74.922,
                            34:78.96,35:79.904,36:83.798,37:85.468,38:87.62,39:88.906,
                            40:91.224,41:92.906,42:95.96,43:98,44:101.07,45:102.906,46:106.42,
                            47:107.868,48:112.411,49:114.818,50:118.71,51:121.76,52:127.6,
                            53:126.904,54:131.293,55:132.905,56:137.327,57:138.905,58:140.116,
                            59:140.908,60:144.242,61:145,62:150.36,63:151.964,64:157.25,65:158.925,
                            66:162.5,67:164.93,68:167.259,69:168.934,70:173.054,71:174.967,72:178.49,
                            73:180.948,74:183.84,75:186.207,76:190.23,77:192.217,78:195.084,
                            79:196.967,80:200.59,81:204.383,82:207.2,83:208.98,84:210,85:210,86:222,
                            87:223,88:226,89:227,90:232.038,91:231.036,92:238.029,93:237,94:244,
                            95:243,96:247,97:247,98:251,99:252,100:257,101:258,102:259,103:262,104:261,105:262,106:266}

def get_atomic_number_from_name(atomic_names:list[str]):
    res = []
    for name in atomic_names:
        res.append(ELEMENTTABLE[name])
    return res

def get_atomic_name_from_number(atomic_number:list[int]):
    res = []
    for number in atomic_number:
        res.append(ELEMENTTABLE_2[number])
    return res

def get_atomic_name_from_str(atom_strs):
    try:
        return [int(_) for _ in atom_strs]
    except ValueError:
        return get_atomic_number_from_name(atom_strs)


# print(get_atomic_name_from_str([8, 72]))

# print(get_atomic_name_from_str(["8", "72"]))

# print(get_atomic_name_from_str(["O", "Hf"]))
