import os
from pwact.active_learning.user_input.scf_param import DFTInput
from pwact.active_learning.user_input.iter_input import SCFParam
from pwact.utils.file_operation import str_list_format
from pwact.utils.json_operation import get_parameter, get_required_parameter
from pwact.utils.constant import DFT_STYLE, PWDATA
from pwact.utils.app_lib.pwmat import read_and_check_etot_input
class InitBulkParam(object):
    def __init__(self, json_dict: dict) -> None:
        self.root_dir = get_parameter("work_dir", json_dict, "./")
        if not os.path.isabs(self.root_dir):
            self.root_dir = os.path.realpath(self.root_dir)
        
        self.data_shuffle = get_parameter("data_shuffle", json_dict, True)
        self.train_valid_ratio = get_parameter("train_valid_ratio", json_dict, 0.8)
        self.interval = get_parameter("interval", json_dict, 1)

        # self.reserve_pwmat_files = get_parameter("reserve_pwmat_files", json_dict, False)
        self.reserve_work = get_parameter("reserve_work", json_dict, False)
        # read init configs
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        if isinstance(sys_configs, dict):
            sys_configs = [sys_configs]

        # set sys_config detail
        self.dft_style = get_required_parameter("dft_style", json_dict).lower()
        self.scf_style = get_parameter("scf_style", json_dict, None)

        self.sys_config:list[Stage] = []
        self.is_relax = False
        self.is_aimd = False
        self.is_scf = False
        for index, config in enumerate(sys_configs):
            stage = Stage(config, index, sys_config_prefix, self.dft_style)
            self.sys_config.append(stage)
            if stage.relax:
                self.is_relax = True
            if stage.aimd:
                self.is_aimd = True
            if stage.scf:
                self.is_scf = True
                
        # for PWmat: set etot.input files and persudo files
        # for Vasp: set INCAR files and persudo files
        self.dft_input = SCFParam(json_dict=json_dict, is_scf=self.is_scf, is_relax=self.is_relax, is_aimd=self.is_aimd, root_dir=self.root_dir, dft_style=self.dft_style, scf_style=self.scf_style)
        # check and set relax etot.input file
        for config in self.sys_config:
            if self.is_relax:
                if config.relax_input_idx >= len(self.dft_input.relax_input_list):
                    raise Exception("Error! for config '{}' 'relax_input_idx' {} not in 'relax_input'!".format(os.path.basename(config.config_file), config.relax_input_idx))
                config.set_relax_input_file(self.dft_input.relax_input_list[config.relax_input_idx])
            if self.is_scf:
                if not os.path.exists(self.dft_input.scf_input_list[0].input_file):
                    raise Exception("Error! relabel dft input file {} not exisit!".format(self.dft_input.scf_input_list[0].input_file))
                config.set_scf_input_file(self.dft_input.scf_input_list[0])
        # check and set aimd etot.input file
        for config in self.sys_config:
            if self.is_aimd:
                if config.aimd_input_idx >= len(self.dft_input.aimd_input_list):
                    raise Exception("Error! for config '{}' 'aimd_input_idx' {} not in 'aimd_input'!".format(os.path.basename(config.config_file), config.aimd_input_idx))
                config.set_aimd_input_file(self.dft_input.aimd_input_list[config.aimd_input_idx])

class Stage(object):
    def __init__(self, json_dict: dict, index:int, sys_config_prefix:str = None, dft_style:str=None) -> None:
        self.dft_style = dft_style #not used
        self.config_index = index
        self.use_dftb = False
        self.use_skf = False
        
        config_file = get_required_parameter("config", json_dict)
        self.config_file = os.path.join(sys_config_prefix, config_file) if sys_config_prefix is not None else config_file
        if not os.path.exists:
            raise Exception("ERROR! The sys_config {} file does not exist!".format(self.config_file))
        self.format = get_parameter("format", json_dict, PWDATA.pwmat_config).lower()
        self.pbc = get_parameter("pbc", json_dict, [1,1,1])
        # extract config file to Config object, then use it
        self.relax = get_parameter("relax", json_dict, True)
        self.relax_input_idx = get_parameter("relax_input_idx", json_dict, 0)
        self.relax_input_file = None
        
        self.aimd = get_parameter("aimd", json_dict, True)
        self.aimd_input_idx = get_parameter("aimd_input_idx", json_dict, 0)
        self.aimd_input_file = None
        
        self.scf = get_parameter("scf", json_dict, False)

        super_cell = get_parameter("super_cell", json_dict, [])
        super_cell = str_list_format(super_cell)
        if len(super_cell) > 0:
            if len(super_cell) == 3 :#should c
                if isinstance(super_cell[0], int):
                    super_cell = [[super_cell[0], 0, 0], [0, super_cell[1], 0], [0, 0, super_cell[2]]]
            elif len(super_cell) == 9:
                super_cell = [super_cell[0:3], super_cell[3:6], super_cell[6:9]]
            else:
                raise Exception("Error! The input super_cell should be 3 or 9 values, for example:\
                    list format [1,2,1], [1,0,0,0,2,0,0,0,3]!")
            self.super_cell = super_cell
        else:
            self.super_cell = None
        
        scale = get_parameter("scale", json_dict, [])
        scale = str_list_format(scale)
        if len(scale) > 0:
            self.scale = scale
        else:
            self.scale = None
            
        self.perturb = get_parameter("perturb", json_dict, 0)
        if self.perturb == 0:
            self.perturb = None
        self.cell_pert_fraction = get_parameter("cell_pert_fraction", json_dict, 0.03)
        self.atom_pert_distance = get_parameter("atom_pert_distance", json_dict, 0.01)
    
    def set_relax_input_file(self, input_file:DFTInput):
        self.relax_input_file = input_file.input_file
        self.relax_kspacing = input_file.kspacing 
        self.relax_flag_symm = input_file.flag_symm

    def set_scf_input_file(self, input_file:DFTInput):
        self.scf_input_file = input_file.input_file
        self.scf_kspacing = input_file.kspacing 
        self.scf_flag_symm = input_file.flag_symm

    def set_aimd_input_file(self, input_file:DFTInput):
        self.aimd_input_file = input_file.input_file
        self.aimd_kspacing = input_file.kspacing
        self.aimd_flag_symm = input_file.flag_symm
        self.use_dftb = input_file.use_dftb
        self.use_skf = input_file.use_skf
