import os
from pwact.utils.constant import DFT_STYLE, CP2K
from pwact.utils.app_lib.pwmat import read_and_check_etot_input
from pwact.utils.json_operation import get_parameter, get_required_parameter
from pwact.utils.app_lib.common import get_vasp_pseudo_atom_type
class SCFParam(object):
    def __init__(self, json_dict:dict, 
        is_relax:bool=False, 
        is_aimd:bool=False, 
        is_scf:bool=False, 
        root_dir:str=None, 
        dft_style:str=None,
        scf_style:str=None) -> None:# for scf relabel in init_bulk
        
        self.dft_style = dft_style
        self.root_dir = root_dir

        self.relax_input_list = []
        self.aimd_input_list = []
        self.scf_input_list = []
        self.pseudo = []
        self.scf_pseudo = []
        self.use_dftb = False

        if is_scf:
            if "scf_input" in json_dict.keys(): # for init_bulk relabel
                json_scf = get_required_parameter("scf_input", json_dict)
                self.scf_input_list = self.set_input(json_scf, flag_symm=0)
            else: # for run_iter
                self.scf_input_list = self.set_input(json_dict, flag_symm=0)
                if self.scf_input_list[0].use_dftb:
                    self.use_dftb = True
        if is_aimd:
            json_aimd = get_required_parameter("aimd_input", json_dict)
            self.aimd_input_list = self.set_input(json_aimd, flag_symm=0)
            if self.aimd_input_list[0].use_dftb:
                self.use_dftb = True
        if is_relax:
            json_relax = get_required_parameter("relax_input", json_dict)
            self.relax_input_list = self.set_input(json_relax, flag_symm=3) 
            if self.relax_input_list[0].use_dftb:
                self.use_dftb = True
        # for pwmat, use 'pseudo' key
        # for vasp is INCAR file, use 'pseudo' key        
        pseudo = get_parameter("pseudo", json_dict, [])
        self.pseudo = self._set_pseudo(pseudo, dft_style)
        
        if is_scf:
            scf_pseudo = get_parameter("scf_pseudo", json_dict, [])
            self.scf_pseudo = self._set_pseudo(scf_pseudo, scf_style)

        # for pwmat-dftb is in_skf, a dir string
        in_skf = get_parameter("in_skf", json_dict, None)
        self.in_skf = None
        if self.use_dftb:
            if in_skf is not None:
                self.in_skf = in_skf if os.path.isabs(in_skf) else os.path.abspath(in_skf)
            #     if not os.path.exists(self.in_skf):
            #         raise Exception("ERROR! The 'in_skf' dir {} not exsit!".format(self.in_skf))
            # else:
            #     raise Exception("ERROR! The 'USE_DFTB' is set in scf.input file, but the 'in_skf' dir not set!")
        # else:
        #     pass
        # for cp2k
        self.basis_set_file = get_parameter("basis_set_file", json_dict, None)
        self.potential_file = get_parameter("potential_file", json_dict, None)

    def _set_pseudo(self, pseudo, style:str):
        res_pseudo = []
        if isinstance(pseudo, str):
            pseudo = list(pseudo)
        for pf in pseudo:
            if not os.path.exists(pf):
                raise Exception("Error! pseduo file {} does not exist!".format(pf))
            if not os.path.isabs(pf):
                pf = os.path.abspath(pf)
            # for vasp pseudo files, read the pseduo files and get the atom type in pseduo
            if style == DFT_STYLE.vasp:
                atom_type = get_vasp_pseudo_atom_type(pf)
            # for pwmat pseudo files, get the atom type from pseudo file name
            elif style == DFT_STYLE.pwmat:
                atom_type = os.path.basename(pf).split('.')[0]
            res_pseudo.append([pf, atom_type])
        return res_pseudo

    '''
    description: 
        set dft input file
        for PWmat: etot.input
        for Vasp: Incar
        for cp2k: input.inp file

        json_input: could be a string or list string, for input control file or a list files
                    could be a dict ir dict list, mainly for pwmat to set kspacing

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
        
        # for vasp incar: the input is a str list of multi incar files
        elif isinstance(json_input, list) and isinstance(json_input[0], str):
            for input_file in json_input:
                if not os.path.isabs(input_file):
                    input_file = os.path.join(self.root_dir, input_file)
                if not os.path.exists(input_file):
                    raise Exception("Error! The input file {} does not exist!".format(input_file))
                input_list.append(DFTInput(input_file, self.dft_style, flag_symm, None))

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

        # check etot input file
        if self.dft_style == DFT_STYLE.pwmat:
            key_values, etot_lines = read_and_check_etot_input(self.input_file)

            if "MP_N123" in key_values and self.kspacing is not None:
                error_info = "ERROR! The 'kspacing' in DFT/input/{} dict and 'MP_N123' in {} file cannot coexist.\n".format(os.path.basename(self.input_file), os.path.basename(self.input_file))
                error_info += "If 'MP_N123' is not indicated in DFT/input/{}, the 'kspacing' param will be used to generate the 'MP_N123' parameter\n".format(os.path.basename(self.input_file))
                raise Exception(error_info)
            elif "MP_N123" not in key_values and self.kspacing is None:
                self.kspacing  = 0.5

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
    