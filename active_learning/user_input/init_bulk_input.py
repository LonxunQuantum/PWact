from active_learning.user_input.param_input import SCFParam

import json
import os

from utils.file_operation import get_required_parameter, get_parameter, str_list_format
from utils.constant import SCF_FILE_STRUCTUR
from utils.app_lib.pwmat import read_and_check_etot_input

class InitBulkParam(object):
    def __init__(self, json_dict: dict) -> None:
        self.root_dir = get_parameter("work_dir", json_dict, "work_dir")
        if not os.path.isabs(self.root_dir):
            slef.root_dir = os.path.realpath(self.root_dir)
        
        self.reserve_scf_files = get_parameter("reserve_scf_files", json_dict, False)

        # read init configs
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        if isinstance(sys_configs, str):
            sys_configs = [sys_configs]

        self.sys_config:list[Stage] = []
        is_relax = False
        is_scf = False
        for index, config in enumerate(sys_configs):
            stage = Stage(config, index, sys_config_prefix)
            self.sys_config.append(stage)
            if stage.relax and is_relax is False:
                is_relax = True
            if stage.AIMD and is_scf is False:
                is_scf = True
        # read scf infos
        self.scf = SCFParam(json_dict["scf"], is_relax, is_scf, self.root_dir)
        

class Stage(object):
    def __init__(self, json_dict: dict, index:int, sys_config_prefix:str = None) -> None:
        self.config_index = index
        config_file = get_required_parameter("config", json_dict)
        self.config = os.path.join(sys_config_prefix, config_file) if sys_config_prefix is not None else config_file
        if not os.path.exists:
            raise Exception("ERROR! The sys_config {} file does not exist!".format(self.config))
        self.relax = get_parameter("relax", json_dict, True)
        
        super_cell = get_parameter("super_cell", json_dict, [])
        super_cell = str_list_format(super_cell)
        if len(super_cell) > 0:
            if len(super_cell) == 3:
                super_cell = [[super_cell[0], 0, 0], [0, super_cell[0], 0], [0, 0, super_cell[0]]]
            elif len(super_cell) == 9:
                pass
            else:
                raise Exception("Error! The input super_cell should be 3 or 9 values, for example:\
                    string format '3 3 3', '1, 2, 3', or '1 2 3 4 5 6 7 8 9'\n\
                    or list format [3 3 3], [1 2 3 4 5 6 7 8 9]!")
            self.super_cell = super_cell
        else:
            self.super_cell = None

        scale = get_parameter("scale", json_dict, [])
        scale = str_list_format(scale)
        self.scale = scale

        self.perturb = get_parameter("perturb", json_dict, None)
        self.cell_pert_fraction = get_parameter("cell_pert_fraction", json_dict, 0.03)
        self.atom_pert_distance = get_parameter("atom_pert_distance", json_dict, 0.01)
    
        self.AIMD = get_parameter("aimd", json_dict, True)
        
        