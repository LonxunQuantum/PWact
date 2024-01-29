import os
from utils.json_operation import get_parameter, get_required_parameter
from utils.constant import MODEL_CMD, FORCEFILED, UNCERTAINTY, PWMAT, DFT_STYLE
from active_learning.user_input.train_param.train_param import TrainParam
from active_learning.user_input.scf_param import SCFParam

class InputParam(object):
    # _instance = None
    def __init__(self, json_dict: dict) -> None:
        self.root_dir = get_parameter("work_dir", json_dict, "work_dir")
        if not os.path.isabs(self.root_dir):
            self.root_dir = os.path.realpath(self.root_dir)
        self.record_file = get_parameter("record_file", json_dict, "{}.record".format(os.path.basename(self.root_dir)))
        print("Warning! record_file not provided, automatically set to {}! ".format(self.record_file))
        
        self.reserve_work = get_parameter("reserve_work", json_dict, False)
        # self.reserve_feature = get_parameter("reserve_feature", json_dict, False)
        self.reserve_md_traj = get_parameter("reserve_md_traj", json_dict, False)
        self.reserve_scf_files = get_parameter("reserve_scf_files", json_dict, False)

        init_data = get_parameter("init_data", json_dict, [])
        self.init_data = self.get_init_data(init_data)
        
        self.train = TrainParam(json_input=json_dict["train"], cmd=MODEL_CMD.train)
        self.strategy = StrategyParam(json_dict["strategy"])
        self.explore = ExploreParam(json_dict["explore"])
        dft_style = get_required_parameter("dft_style", json_dict["dft"])
        self.scf = SCFParam(json_dict=json_dict["dft"], dft_style=dft_style, is_scf=True, root_dir = self.root_dir)

    def to_dict(self):
        res = {}
        res["work_dir"] = self.root_dir
        res["record_file"] = self.record_file
        
        res["reserve_feature"] = self.reserve_feature
        res["reserve_md_traj"] = self.reserve_md_traj
        res["reserve_scf_files"] = self.reserve_scf_files
        
        res["train"] = self.train.to_dict()
        res["strategy"] = self.strategy.to_dict()

        return res

    def get_init_data(self, init_data:list[str]):
        init_data_path = []
        for _data in init_data:
            _data_path = _data if os.path.isabs(_data) else os.path.join(self.root_dir, _data)
            if not os.path.exists(_data_path):
                print("Warning! the init data {} does not exist!".format(_data_path))
                continue
            init_data_path.append(_data_path)
        return init_data_path
        
class StrategyParam(object):
    def __init__(self, json_dict) -> None:
        self.md_type = get_parameter("md_type", json_dict, FORCEFILED.libtorch_lmps)
        
        self.max_select = get_parameter("max_select", json_dict, 1000)
        self.uncertainty = get_parameter("uncertainty", json_dict, UNCERTAINTY.committee)
        if self.uncertainty.upper() == UNCERTAINTY.kpu:
            self.model_num = 1
            self.base_kpu_max_images = get_parameter("base_kpu_max_images", json_dict, 200)
            self.base_kpu_mvm_ratio = get_parameter("base_kpu_mvm_ratio", json_dict, 0.2)
            self.kpu_upper = get_parameter("kpu_upper", json_dict, 1.5)
            self.kpu_lower = get_parameter("kpu_lower", json_dict, 0.5)
        elif self.uncertainty.upper() == UNCERTAINTY.committee:
            self.model_num = get_parameter("model_num", json_dict, 4)
            self.lower_model_deiv_f = get_required_parameter("lower_model_deiv_f", json_dict)
            self.upper_model_deiv_f = get_required_parameter("upper_model_deiv_f", json_dict)
        else:
            raise Exception("ERROR! uncertainty strategy {} not support! Please check!".format(self.uncertainty))

        self.compress = get_parameter("compress", json_dict, False)
        self.compress_order = get_parameter("compress_order", json_dict, 3)
        self.compress_dx = get_parameter("compress_dx", json_dict, 0.01)
        
        if self.compress:
            if self.uncertainty == UNCERTAINTY.committee and self.md_type == FORCEFILED.fortran_lmps:
                raise Exception("Error! The compress model does not fortran lammps! Please set the 'md_type' to 2!")
                    
    def to_dict(self):
        res = {}
        res["md_type"] = self.md_type
        res["max_select"] = self.max_select
        res["uncertainty"] = self.uncertainty
        res["model_num"] = self.model_num

class SysConfig(object):
    def __init__(self, sys_config:str, format:str) -> None:
        self.sys_config = sys_config
        self.format = format

class ExploreParam(object):
    def __init__(self, json_dict) -> None:
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        if isinstance(sys_configs, str) or isinstance(sys_configs, dict):
            sys_configs = [sys_configs]
        self.sys_configs:list[SysConfig]=[]
        for sys_config in sys_configs:
            if isinstance(sys_config, str):
                config = os.path.join(sys_config_prefix, sys_config) if sys_config_prefix is not None else sys_config
                config_format = DFT_STYLE.pwmat
            elif isinstance(sys_config, dict):
                config = os.path.join(sys_config_prefix, sys_config["config"]) if sys_config_prefix is not None else sys_config["config"]
                config_format = get_parameter("format", sys_config, DFT_STYLE.pwmat)  
            if not os.path.exists:
                raise Exception("ERROR! The sys_config {} file does not exist!".format(config))
            self.sys_configs.append(SysConfig(config, config_format))
        
        # set md deatils
        self.md_job_list = self.set_md_details(json_dict["md_jobs"])
        self.md_job_num = len(self.md_job_list)

    def set_md_details(self, md_list_dict:list[dict]):
        iter_md:list[list[MdDetail]] = []
        for iter_index, md_dict in enumerate(md_list_dict): # for each iter
            iter_exp_md:list[MdDetail] = []
            if not isinstance(md_dict, list):
                md_dict = [md_dict]
            for md_exp_id, md_exp in enumerate(md_dict):
                iter_exp_md.append(MdDetail(md_exp_id, md_exp))
            iter_md.append(iter_exp_md)
        return iter_md
    
    def to_dict(self):
        res = {}
        return res

class MdDetail(object):
    def __init__(self, md_index: int, json_dict:dict) -> None:
        self.md_index = md_index
        self.nsteps = get_required_parameter("nsteps", json_dict)
        self.md_dt = get_parameter("md_dt", json_dict, 0.001) #fs
        self.trj_freq = get_parameter("trj_freq", json_dict, 10)
        
        self.ensemble = get_parameter("ensemble", json_dict, "nvt")
        
        self.press_list = get_parameter("press", json_dict, [])
        self.taup = get_parameter("taup", json_dict, 0.5)
        if not isinstance(self.press_list, list):
            self.press_list = [self.press_list]
        self.temp_list = get_parameter("temps", json_dict, [])
        self.taut = get_parameter("taut", json_dict, 0.1)
        
        if not isinstance(self.temp_list, list):
            self.temp_list = [self.temp_list]
            
        self.sys_idx = get_required_parameter("sys_idx", json_dict)
        if not isinstance(self.sys_idx, list):
            self.sys_idx = [self.sys_idx]
        self.kspacing = get_parameter("temps", json_dict, None)
        self.neigh_modify = get_parameter("neigh_modify", json_dict, 10)
        self.mass = get_parameter("mass",json_dict, None)
        self.merge_traj = get_parameter("merge_traj", json_dict, False)
        self.boundary = get_parameter("boundary", json_dict, True)
    '''
    description: 
    maybe not use
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_md_combination(self):
        md_list = []
        for t, temp in enumerate(self.temp_list):
            for p, press in enumerate(self.press_list):
               md_list.append(t, p) 
        
