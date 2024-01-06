import json
import os

from utils.file_operation import get_required_parameter, get_parameter
from utils.constant import ENSEMBLE, TRAIN_INPUT_PARAM, FORCEFILED, UNCERTAINTY, SCF_FILE_STRUCTUR
from utils.app_lib.pwmat import read_and_check_etot_input
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

        init_mvm_files = get_parameter("init_mvm_files", json_dict, [])
        
        self.train = TrainParam(json_dict["train"], self.root_dir, init_mvm_files)
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
    def __init__(self, json_dict:dict, root_dir:str, init_mvm_files:list[str]) -> None:
        self.root_dir = root_dir
        self.data_retrain = get_parameter("data_retrain", json_dict, 20)
        self.init_mvm_files = self.get_init_mvm_files(init_mvm_files)
        self.train_input_file = get_required_parameter("train_input_file", json_dict)
        if not os.path.isabs(self.train_input_file):
            self.train_input_file = os.path.join(self.root_dir, self.train_input_file)
        if not os.path.exists(self.train_input_file):
            raise Exception("Error! The {} file not exists!".format(self.train_input_file))
        self.train_input_dict:dict = json.load(open(self.train_input_file))
        # is type_embedding
        if TRAIN_INPUT_PARAM.type_embedding in self.train_input_dict.keys() or \
            "model" in self.train_input_dict.keys() and TRAIN_INPUT_PARAM.type_embedding in train_input_dict["model"].keys:
                self.type_embedding = True
        else:
            self.type_embedding = False
        # model_type
        self.model_type = self.train_input_dict[TRAIN_INPUT_PARAM.model_type]
        # atom_type
        self.atom_type = self.train_input_dict[TRAIN_INPUT_PARAM.atom_type]
        
    def get_init_mvm_files(self, init_mvm_files:list[str]):
        init_file_path = []
        for mvm in init_mvm_files:
            mvm_path = mvm if os.path.isabs(mvm) else os.path.join(self.root_dir, mvm)
            if not os.path.exists(mvm_path):
                print("Warning! the init_mvm_file {} does not exist!".format(mvm))
                continue
            init_file_path.append(mvm_path)
        return init_file_path

    def get_train_input_dict(self):
        train_input_dict = self.train_input_dict.copy()
        return train_input_dict

    def to_dict(self):
        return self.train_input_dict

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
        
class ExploreParam(object):
    def __init__(self, json_dict) -> None:
        sys_config_prefix = get_parameter("sys_config_prefix", json_dict, None)
        sys_configs = get_required_parameter("sys_configs", json_dict)
        self.sys_configs = []
        for sys_config in sys_configs:
            config = os.path.join(sys_config_prefix, sys_config) if sys_config_prefix is not None else sys_config
            if not os.path.exists:
                raise Exception("ERROR! The sys_config {} file does not exist!".format(config))
            self.sys_configs.append(config)
        
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
        md_traj_id = 0
        for t, temp in enumerate(self.temp_list):
            for p, press in enumerate(self.press_list):
               md_list.append(t, p) 
        
class SCFParam(object):
    def __init__(self, json_dict) -> None:
        self.etot_input_file = get_parameter("etot_input_file", json_dict, None)
        self.pseudo = get_required_parameter("pseudo", json_dict)
        if isinstance(self.pseudo, str):
            self.pseudo = list(self.pseudo)
        self.__init_variable()
        if self.etot_input_file is None:
            self.set_etot_input_detail(json_dict)
            self.etot_input_content = None
        else:
            if not os.path.exists(self.etot_input_file):
                raise Exception("the input etot.input file {} dest not exist!".format(self.etot_input_file))
            self.etot_input_content = read_and_check_etot_input(self.etot_input_file)

    @staticmethod
    def get_pseudo_by_atom_name(pseduo_list:list[str], atom_name):
        for pseduo in pseduo_list:
            if atom_name in pseduo:
                return pseduo
        return None
    
    def __init_variable(self):
        self.node1 = None
        self.node2 = None
        self.e_error = None
        self.rho_error = None
        self.ecut = None
        self.ecut2 = None
        self.kspacing = None
        self.out_wg = None
        self.out_rho = None
        self.out = None
        self.out_force = None
        self.out_stress = None
        self.out_mlmd = None
        self.MP_N123 = None
        self.SCF_ITER0_1 = None
        self.SCF_ITER0_2 = None
        self.energy_decomp = None
        self.energy_decomp_special2 = None
        self.flag_symm = None
        self.icmix = None
        self.smearing = None
        self.sigma = None

    def set_etot_input_detail(self, json_dict):
        self.node1 = get_required_parameter("node1", json_dict, 1)
        self.node2 = get_required_parameter("node2", json_dict, 4)
        
        self.e_error = get_parameter("e_error", json_dict,  1.0e-6)
        self.rho_error = get_parameter("rho_error", json_dict,  1.0e-4)
        self.ecut = get_required_parameter("ecut", json_dict)
        self.ecut2 = get_parameter("ecut2", json_dict,  self.ecut*4)
        
        self.kspacing = get_parameter("kspacing", json_dict, 0.5)
        
        self.out_wg = get_parameter("out_wg", json_dict, "F")
        self.out_rho = get_parameter("out_rho", json_dict, "F")
        self.out = get_parameter("out.vr", json_dict, "F")
        self.out_force = get_parameter("out_force", json_dict, "T")
        self.out_stress = get_parameter("out_stress", json_dict, "T")
        self.out_mlmd = get_parameter("out_mlmd", json_dict, "F")
        self.MP_N123 = get_parameter("MP_N123", json_dict, None) #MP_N123 is None then using 'kespacing' generates it
        self.SCF_ITER0_1 = get_parameter("SCF_ITER0_1", json_dict,  None)
        self.SCF_ITER0_2 = get_parameter("SCF_ITER0_2", json_dict,  None)
        self.energy_decomp = get_parameter("energy_decomp", json_dict,  "T")
        self.energy_decomp_special2 = get_parameter("energy_decomp_special2", json_dict,  "2, 0.05, 1.5")
        self.flag_symm = get_parameter("flag_symm", json_dict,None)
        self.icmix = get_parameter("icmix", json_dict, None)
        self.smearing = get_parameter("smearing", json_dict, None)
        self.sigma = get_parameter("sigma", json_dict, None)

        