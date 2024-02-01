import os
from utils.constant import DFT_STYLE, DFT_TYPE
from utils.app_lib.pwmat import read_and_check_etot_input
# from utils.app_lib.vasp import read_and_check_incar
from utils.json_operation import get_parameter, get_required_parameter
from utils.app_lib.common import get_vasp_pseudo_atom_type
class SCFParam(object):
    def __init__(self, json_dict:dict, 
        is_relax:bool=False, 
        is_aimd:bool=False, 
        is_scf:bool=False, 
        root_dir:str=None, 
        dft_style:str=None) -> None:
        
        self.dft_style = dft_style
        self.root_dir = root_dir
        if is_scf:
            self.scf_input_list = self.set_input(json_dict, flag_symm=0)
        if is_aimd:
            json_aimd = get_required_parameter("aimd_input", json_dict)
            self.aimd_input_list = self.set_input(json_aimd, flag_symm=0)
        if is_relax:
            json_relax = get_required_parameter("relax_input", json_dict)
            self.relax_input_list = self.set_input(json_relax, flag_symm=3) 

        if is_scf or is_relax or (is_aimd and self.aimd_input_list[0].use_dftb is False):
            pseudo = get_required_parameter("pseudo", json_dict)
        else:
            pseudo = []
        if isinstance(pseudo, str):
            pseudo = list(pseudo)
        self.pseudo = []
        for pf in pseudo:
            if not os.path.exists(pf):
                raise Exception("Error! pseduo file {} does not exist!".format(pf))
            # for vasp pseudo files, read the pseduo files and get the atom type in pseduo
            if self.dft_style == DFT_STYLE.vasp:
                atom_type = get_vasp_pseudo_atom_type(pf)
            # for pwmat pseudo files, get the atom type from pseudo file name
            elif self.dft_style == DFT_STYLE.pwmat:
                atom_type = os.path.basename(pf).split('.')[0]
            self.pseudo.append([pf, atom_type])
        
        self.in_skf = None
        if is_aimd and self.aimd_input_list[0].use_dftb and self.aimd_input_list[0].use_skf:
            IN_SKF = get_required_parameter("in_skf", json_dict)
            self.in_skf = IN_SKF if os.path.isabs(IN_SKF) else os.path.abspath(IN_SKF)

    '''
    description: 
        set dft input file
        for PWmat: etot.input
        for Vasp: Incar
        json_input: could be a dict or a list of dict for pwamt etot.input files
                    or a string or string list for Incar file path
    param {*} self
    param {*} json_etot
    param {*} root_dir
    param {int} flag_symm
    return {*}
    author: wuxingxing
    '''    
    def set_input(self, json_input, flag_symm:int):
        input_list = []
        # for vasp incar: the input is a str
        if isinstance(json_input, str):
            input_file = json_input
            if not os.path.isabs(input_file):
                input_file = os.path.join(self.root_dir, input_file)
            if not os.path.exists(input_file):
                raise Exception("Error! The input file {} does not exist!".format(input_file))
            input_list = [DFTInput(input_file, self.dft_style, flag_symm, None)]
        # for vasp incar: the input is a str list of multi incar files
        elif isinstance(json_input, list) and isinstance(json_input[0], str):
            for input_file in json_input:
                if not os.path.isabs(input_file):
                    input_file = os.path.join(self.root_dir, input_file)
                if not os.path.exists(input_file):
                    raise Exception("Error! The input file {} does not exist!".format(input_file))
                input_list.append(DFTInput(input_file, self.dft_style, flag_symm, None))

        # for pwmat etot.input: the input is a dict including etot.input file path, kspacing and flag_symm
        elif isinstance(json_input, dict):
            input_file = get_required_parameter("input", json_input)
            if not os.path.isabs(input_file):
                input_file = os.path.join(self.root_dir, input_file)
            if not os.path.exists(input_file):
                raise Exception("Error! The input file {} does not exist!".format(input_file))
            flag_symm = get_parameter("flag_symm", json_input, flag_symm)
            kspacing = get_parameter("kspacing", json_input, None)
            input_list = [DFTInput(input_file, self.dft_style, flag_symm, kspacing)]
        # for pwmat etot.input: the input is a dict list: for each dict including etot.input file path, kspacing and flag_symm
        elif isinstance(json_input, list) and isinstance(json_input[0], dict):
            for json_detail in json_input:
                input_file = get_required_parameter("input", json_detail)
                if not os.path.isabs(input_file):
                    input_file = os.path.join(self.root_dir, input_file)
                if not os.path.exists(input_file):
                    raise Exception("Error! The etot.input file {} does not exist!".format(input_file))
                flag_symm = get_parameter("flag_symm", json_detail, flag_symm)
                kspacing = get_parameter("kspacing", json_detail, None)
                input_list.append(DFTInput(input_file, self.dft_style, flag_symm, kspacing))
        else:
            raise Exception("the dft input file cat not recognized!")

        return input_list

class DFTInput(object):
    def __init__(self, input_file:str, dft_style:str, flag_symm:int, kspacing:int) -> None:
        # super().__init__(input_file=input_file, dft_style=dft_style)
        self.input_file = input_file
        self.dft_style = dft_style
        self.kspacing = kspacing
        self.flag_symm = flag_symm
        self.use_dftb = False
        self.use_skf = False
        if self.kspacing is None:
            self.kspacing_default = 0.5
        # check etot input file
        if self.dft_style == DFT_STYLE.pwmat:
            key_values, etot_lines = read_and_check_etot_input(self.input_file)
            if "USE_DFTB" in key_values.keys() \
                and key_values["USE_DFTB"] is not None \
                    and key_values["USE_DFTB"] == "T":
                self.use_dftb = True
                if key_values["DFTB_DETAIL"].replace(",", " ").split()[0] != "3": # not chardb
                    self.use_skf = True
            
    
    def get_input_content(self):
        if self.dft_style == DFT_STYLE.pwmat:
            return read_and_check_etot_input(self.input_file)
        elif self.dft_style == DFT_STYLE.vasp:
            with open(self.input_file, "r") as fp:
                lines = fp.readlines()
            return lines
        else:
            raise Exception("the dft style {} not realized!".format(self.dft_style))    
    
    # these code for etot.input script generated from user input param
    # def __init_variable(self):
    #     self.node1 = None
    #     self.node2 = None
    #     self.e_error = None
    #     self.rho_error = None
    #     self.ecut = None
    #     self.ecut2 = None
    #     self.kspacing = None
    #     self.out_wg = None
    #     self.out_rho = None
    #     self.out = None
    #     self.out_force = None
    #     self.out_stress = None
    #     self.out_mlmd = None
    #     self.MP_N123 = None
    #     self.SCF_ITER0_1 = None
    #     self.SCF_ITER0_2 = None
    #     self.energy_decomp = None
    #     self.energy_decomp_special2 = None
    #     self.flag_symm = None
    #     self.icmix = None
    #     self.smearing = None
    #     self.sigma = None

    # def set_etot_input_detail(self, json_dict):
        # self.node1 = get_required_parameter("node1", json_dict, 1)
        # self.node2 = get_required_parameter("node2", json_dict, 4)
        
        # self.e_error = get_parameter("e_error", json_dict,  1.0e-6)
        # self.rho_error = get_parameter("rho_error", json_dict,  1.0e-4)
        # self.ecut = get_required_parameter("ecut", json_dict)
        # self.ecut2 = get_parameter("ecut2", json_dict,  self.ecut*4)
        
        # self.kspacing = get_parameter("kspacing", json_dict, 0.5)
        
        # self.out_wg = get_parameter("out_wg", json_dict, "F")
        # self.out_rho = get_parameter("out_rho", json_dict, "F")
        # self.out = get_parameter("out.vr", json_dict, "F")
        # self.out_force = get_parameter("out_force", json_dict, "T")
        # self.out_stress = get_parameter("out_stress", json_dict, "T")
        # self.out_mlmd = get_parameter("out_mlmd", json_dict, "F")
        # self.MP_N123 = get_parameter("MP_N123", json_dict, None) #MP_N123 is None then using 'kespacing' generates it
        # self.SCF_ITER0_1 = get_parameter("SCF_ITER0_1", json_dict,  None)
        # self.SCF_ITER0_2 = get_parameter("SCF_ITER0_2", json_dict,  None)
        # self.energy_decomp = get_parameter("energy_decomp", json_dict,  "T")
        # self.energy_decomp_special2 = get_parameter("energy_decomp_special2", json_dict,  "2, 0.05, 1.5")
        # self.flag_symm = get_parameter("flag_symm", json_dict,None)
        # self.icmix = get_parameter("icmix", json_dict, None)
        # self.smearing = get_parameter("smearing", json_dict, None)
        # self.sigma = get_parameter("sigma", json_dict, None)
        # self.relax_detail = get_parameter("relax_detail", json_dict, None)
        # self.vdw = get_parameter("vdw", json_dict, None)