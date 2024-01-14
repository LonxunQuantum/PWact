from active_learning.user_input.param_input import SCFParam, EtotInput

import os

from utils.file_operation import get_required_parameter, get_parameter, str_list_format

class InitBulkParam(object):
    def __init__(self, json_dict: dict) -> None:
        self.root_dir = get_parameter("work_dir", json_dict, "work_dir")
        if not os.path.isabs(self.root_dir):
            self.root_dir = os.path.realpath(self.root_dir)
        
        self.reserve_scf_files = get_parameter("reserve_scf_files", json_dict, False)

        # read init configs
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        if isinstance(sys_configs, str):
            sys_configs = [sys_configs]

        # set sys_config detail
        self.sys_config:list[Stage] = []
        is_relax = False
        is_aimd = False
        for index, config in enumerate(sys_configs):
            stage = Stage(config, index, sys_config_prefix)
            self.sys_config.append(stage)
            if stage.relax and is_relax is False:
                is_relax = True
            if stage.aimd and is_aimd is False:
                is_aimd = True
                
        # set etot.input files and persudo files
        self.etot_input = SCFParam(json_dict=json_dict, is_relax=is_relax, is_aimd=is_aimd, root_dir=self.root_dir)
        # check and set etot.input file
        for config in self.sys_config:
            if config.relax_etot_idx >= len(self.etot_input.relax_etot_input_list):
                raise Exception("Error! for config '{}' 'relax_etot_idx' {} not in 'relax_etot_input'!".format(os.path.basename(config.config), config.relax_etot_idx))
            if is_relax:
                config.set_relax_etot_input_file(self.etot_input.relax_etot_input_list[config.relax_etot_idx])
            if is_aimd:
                config.set_aimd_etot_input_file(self.etot_input.relax_etot_input_list[config.relax_etot_idx])

class Stage(object):
    def __init__(self, json_dict: dict, index:int, sys_config_prefix:str = None) -> None:
        self.config_index = index
        config_file = get_required_parameter("config", json_dict)
        self.config = os.path.join(sys_config_prefix, config_file) if sys_config_prefix is not None else config_file
        if not os.path.exists:
            raise Exception("ERROR! The sys_config {} file does not exist!".format(self.config))
        
        self.relax = get_parameter("relax", json_dict, True)
        self.relax_etot_idx = get_parameter("relax_etot_idx", json_dict, 0)
        self.relax_etot_file = None
        
        self.aimd = get_parameter("aimd", json_dict, True)
        self.aimd_etot_idx = get_parameter("aimd_etot_idx", json_dict, 0)
        self.aimd_etot_file = None

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
            
        self.perturb = get_parameter("perturb", json_dict, 3)
        if self.perturb == 0:
            self.perturb = None
        self.cell_pert_fraction = get_parameter("cell_pert_fraction", json_dict, 0.03)
        self.atom_pert_distance = get_parameter("atom_pert_distance", json_dict, 0.01)
    
    def set_relax_etot_input_file(self, etot_input:EtotInput):
        self.relax_etot_file = etot_input.etot_input
        self.relax_etot_input = etot_input.kspacing 
        self.relax_etot_input = etot_input.flag_symm

    def set_aimd_etot_input_file(self, etot_input:EtotInput):
        self.aimd_etot_file = etot_input.etot_input
        self.aimd_etot_input = etot_input.kspacing
        self.aimd_etot_input = etot_input.flag_symm