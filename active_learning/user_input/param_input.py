import json
import os

from utils.file_operation import get_required_parameter, get_parameter
from utils.constant import ENSEMBLE
class InputParam(object):
    # _instance = None

    def __init__(self, json_dict: dict) -> None:
        self.root_dir = get_parameter("work_dir", json_dict, "work_dir")
        if not os.path.isabs(self.root_dir):
            slef.root_dir = os.path.realpath(self.root_dir)
        self.record_file = get_parameter("record_file", json_dict, "{}.record".format(os.path.basename(self.root_dir)))
        print("Warning! record_file not provided, automatically set to {}! ".format(self.record_file))
        
        self.reserve_feature = get_parameter("reserve_feature", json_dict, False)
        self.reserve_md_traj = get_parameter("reserve_md_traj", json_dict, False)
        self.reserve_scf_files = get_parameter("reserve_scf_files", json_dict, False)
    
        self.train = TrainParam(json_dict["train"], self.root_dir)
        self.strategy = StrategyParam(json_dict["strategy"])
        self.explore = ExploreParam(json_dict["explore"])
        self.scf = SCFParam(json_dict["scf"])

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
        
    # @classmethod
    # def get_instance(cls, json_dict:dict = None):
    #     if not cls._instance:
    #         cls._instance = cls(json_dict)
    #     return cls._instance

class TrainParam(object):
    def __init__(self, json_dict:dict, root_dir:str) -> None:
        self.root_dir = root_dir
        self.train_input_file = get_parameter("train_file", json_dict, None)
        self.data_retrain = get_parameter("data_retrain", json_dict, 20)

        init_mvm_files = get_parameter("init_mvm_files", json_dict, [])
        self.init_mvm_files = self.get_init_mvm_files(init_mvm_files)
        if self.train_input_file is not None and os.path.exists(self.train_input_file):
            self.train_input_dict = json.load(open(self.train_input_file))
            self.base_input_param = self.get_base_train_input_param(train_input_dict)
        else:
            self.train_input_dict = None
            self.base_input_param = self.get_base_train_input_param(json_dict)
    
    def get_init_mvm_files(self, init_mvm_files:list[str]):
        init_file_path = []
        for mvm in init_mvm_files:
            mvm_path = mvm if os.path.isabs(mvm) else os.path.join(self.root_dir, mvm)
            if not os.path.exists(mvm_path):
                print("Warning! the init_mvm_file {} does not exist!".format(mvm))
                continue
            init_file_path.append(mvm_path)
        return init_file_path

    def get_base_train_input_param(self, json_dict:dict):
        base_input_param = {}
        base_input_param["atom_type"] = get_required_parameter("atom_type", json_dict)
        base_input_param["maxNeighborNum"] = get_required_parameter("maxNeighborNum", json_dict)
        base_input_param["model_type"] = get_required_parameter("model_type", json_dict)
        return base_input_param

    def to_dict(self):
        res = {}
        res["train_file"] = self.train_input_file
        res["data_retrain"] = self.data_retrain
        res["init_mvm_files"] = self.init_mvm_files
        res["atom_type"] = self.base_input_param["atom_type"]
        res["maxNeighborNum"] = self.base_input_param["maxNeighborNum"]
        res["model_type"] = self.base_input_param["model_type"]
        return res

class StrategyParam(object):
    def __init__(self, json_dict) -> None:
        self.md_type = get_parameter("md_type", json_dict, 1)
        
        self.max_select = get_parameter("max_select", json_dict, 100)
        self.uncertainty = get_parameter("uncertainty", json_dict, "KPU", "upper")
        if self.uncertainty == "KPU".upper():
            self.model_num = 1
        elif self.uncertainty == "committee".upper():
            self.model_num = get_parameter("model_num", json_dict, 4)
        else:
            raise Exception("ERROR! uncertainty strategy {} not support! Please check!".format(self.uncertainty))

    def to_dict(self):
        res = {}
        res["md_type"] = self.md_type
        res["max_select"] = self.max_select
        res["uncertainty"] = self.uncertainty
        res["model_num"] = self.model_num
        
class ExploreParam(object):
    def __init__(self, json_dict) -> None:
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        self.sys_configs = []
        for sys_config in sys_configs:
            config = os.path.join(sys_config_prefix, sys_config) if sys_config_prefix is not None else sys_config
            if not os.path.exists:
                raise Exception("ERROR! The sys_config {} file does not exist!".format(config))
            self.sys_configs.appned(config)
        
        # set md deatils
        self.md_details = self.set_md_details(json_dict["sys_configs"])
        self.md_job_num = len(json_dict["md_jobs"])

    def set_md_details(self, json_dict:str):
        pass
    
    def to_dict(self):
        res = {}
        return res

class MdDetail(object):
    def __init__(self, md_index: int, json_dict:dict) -> None:
        self.md_index = md_index
        self.ensemble = get_parameter("ensemble", json_dict, "nvt")
        self.press_list = get_parameter()
          
class SCFParam(object):
    def __init__(self, json_dict) -> None:
        pass