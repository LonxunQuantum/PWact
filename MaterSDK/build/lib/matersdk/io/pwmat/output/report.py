import os
import math
import linecache
import numpy as np
from typing import Dict

from .outfermi import OutFermi


class Report(object):
    COLUMN_PER_LINE = 5 # REPORT中每行出现几个 eigen energies
    
    def __init__(self, report_path:str):
        self.report_path = report_path
    
    
    def _search_aim(self, aim_content:str):
        '''
        Description
        -----------
            1. 查询REPORT文件中是否存在特定的内容(aim_content)，并确定所在的行数
        
        Parameters
        ----------
            1. aim_content: str
                - 特定的内容
        
        Return
        ------
            1. idxs_lst: List[int]
        '''
        with open(self.report_path, "r") as f:
            lines_lst = f.readlines()
        
        idxs_lst = []
        for tmp_idx, tmp_line in enumerate(lines_lst, 1):
            if aim_content in tmp_line:
                idxs_lst.append(tmp_idx)
        
        return idxs_lst
    
    
    def get_num_bands(self):
        '''
        Description
        -----------
            1. 得到能带数 (每个kpoint的本征态)
        
        Return
        ------
            1. num_bands: int
                - 能带的数目
        '''
        ### Step 1. 查询得到 `NUM_BAND` 所在的行
        aim_content = "NUM_BAND"
        idx_num_bands = self._search_aim(aim_content=aim_content)[0]
        #print(idx_num_bands)
        ### Step 2. 提取能带的数目
        specific_line = linecache.getline(self.report_path, idx_num_bands)
        num_bands = int( specific_line.split()[-1] )
        
        return num_bands
    
    
    def get_num_kpts(self):
        '''
        Description
        -----------
            1. 得到Kpoint的数目
        
        Return
        ------
            1. num_kpts: int
                - kpoints 的数目
        '''    
        ### Step 1. 查询得到 `NUM_KPT` 所在的行
        aim_content = "NUM_KPT"
        idx_num_kpts = self._search_aim(aim_content=aim_content)[0]
        
        ### Step 2. 提取kpoints的数目
        specific_line = linecache.getline(self.report_path, idx_num_kpts)
        num_kpts = int( specific_line.split()[-1] )
        
        return num_kpts
        
    
    def get_eigen_energies(self):
        '''
        Description
        -----------
            1. 得到的本征值未减去费米能级
        
        Return
        ------
            1. spin2eigen_energies: Dict[str, np.ndarray]
                - np.ndarray 一维: kpoint
                - np.ndarray 二维: 某kpoint的本征能量
        '''
        ### Step 1. 
        ###     1. 初始化 `spin2eigen_energies`
        ###     2. 得到kpoints的数目 `num_kpts`
        ###     3. 得到bands的数目 `num_bands`
        ###     4. 得到 `idx_eigen_start_lst`
        spin2eigen_energies = {"up":[], "down":[]}
        num_kpts = self.get_num_kpts()
        num_bands = self.get_num_bands()
        aim_content_eigen = "eigen energies, in eV"
        idxs_eigen_start_lst = self._search_aim(
                                aim_content=aim_content_eigen)
        
        ### Step 2. 读取 REPORT 文件
        with open(self.report_path, "r") as f:
            lines_lst = f.readlines()
        
        ### Step 3. 得到每个kpoint的本征能量
        num_lines_for_band = int( np.ceil(num_bands / self.COLUMN_PER_LINE) )
        for tmp_idx, tmp_idx_eigen_start in enumerate(idxs_eigen_start_lst):
            tmp_eigen_energies_ = lines_lst[tmp_idx_eigen_start : tmp_idx_eigen_start+num_lines_for_band]
            tmp_eigen_energies = [float(eigen) for tmp_5_eigen in tmp_eigen_energies_ for eigen in tmp_5_eigen.split()]
            tmp_eigen_energies_array = np.array( tmp_eigen_energies )
            
            if tmp_idx < num_kpts:
                spin2eigen_energies["up"].append(tmp_eigen_energies_array)
            else:
                spin2eigen_energies["down"].append(tmp_eigen_energies_array)
        
        ### Step 4. 将 spin2igen_energies 的 values 变为 np.ndarray 形式
        spin2eigen_energies.update(
                        {"up": np.array( spin2eigen_energies["up"] )}
                        )
        spin2eigen_energies.update(
                        {"down": np.array( spin2eigen_energies["down"] )}
                        )
        
        ### Step 5. 当 ispin 打开时，自旋向上和向下的(kpoints, eigen_states)应该相等
        if spin2eigen_energies["down"].size != 0:
            assert (spin2eigen_energies["up"].shape != spin2eigen_energies["down"])
        
        return spin2eigen_energies
    
    
    def get_in_atom(self):
        '''
        Description
        -----------
            1. 得到输入的结构
        
        Return
        ------
            1. in_atom_name: str
                - basename
                - e.g. "atom.config"
        '''        
        aim_conetent_inatom = "IN.ATOM"
        idx_inatom = self._search_aim(aim_content=aim_conetent_inatom)[0]
        in_atom_name = linecache.getline(self.report_path, idx_inatom).split()[-1]
    
        return in_atom_name
    
    
    def _is_metal(self, 
                out_fermi_path:str,
                efermi_tol:float=1e-4,
                ):
        '''
        Description
        -----------
            1. Check if the bandstructure indicates a metal by looking if the fermi 
            level crosses a band.
        
        Parameters
        ----------
            1. out_fermi_path: str
                - OUT.FERMI 的路径
            2. efermi_tol: float
                - The tolerance of fermi level
        
        Return
        ------
            1. mark: bool
                - True: 是金属
                - False: 不是金属
        '''
        ### Step 1. 得到费米能级
        ### Step 1.1. 判断是否存在 OUT.FERMI 文件
        if not os.path.exists(out_fermi_path):
            raise("当前目录下不存在 OUT.FERMI 文件，无法读取费米能级！")
        ### Step 1.2. 从 OUT.FERMI 中读取费米能级
        out_fermi_object = OutFermi(out_fermi_path=out_fermi_path)
        efermi_ev = out_fermi_object.get_efermi()
        
        ### Step 2. 判断是否有能带穿过费米能级 (check if the fermi level crosses a band)
        spin2eigen_energies:Dict[str, np.ndarray] = self.get_eigen_energies()
        for tmp_spin in list( spin2eigen_energies.keys() ): # ["up", "down"]
            ### engen_energies_T
            ###     - 一维：
            ###     - 二维：
            eigen_energis_T = spin2eigen_energies[tmp_spin].T
            for idx_band in range(eigen_energis_T.shape[0]):
                if np.any(eigen_energis_T[idx_band, :] - efermi_ev < -efermi_tol) and \
                    np.any(eigen_energis_T[idx_band, :] - efermi_ev > efermi_tol):
                    return True
        return False
    
    
    def get_cbm(self, out_fermi_path:str):
        '''
        Description
        -----------
            1. 得到导带顶的 idx_kpt, idx_band, idx_spin, energy
        
        Return
        ------
            1. Union[Dict, None]
            2. cbm_dict: { "energies": List[float], "spins": List[str], "kpts": List[int], bands: List[int] }
                - "energies": 
                - "spins": 
                - "kpts": 
                - "bands": 
                - 是列表形式，因为有时候会共享 cbm
                - e.g.  {'energies': [-0.3426], 'kpts': [19], 'bands': [53], 'spins': ['up']}
            3. 当体系是金属的时候返回 None
            
        Note
        ----
            1. idx_kpt 与 idx_band 均是从 1 开始的 (REPORT的输出信息就是从 1 开始的)
        '''
        ### Step 1. 判断体系是否是半导体
        if self._is_metal(out_fermi_path=out_fermi_path):
            #print("本材料体系是金属")
            #raise SystemExit
            return None
        
        ### Step 2. 得到体系的费米能级
        out_fermi_object = OutFermi(out_fermi_path=out_fermi_path)
        efermi_ev = out_fermi_object.get_efermi()
        
        ### Step 3. 得到本征能量
        spin2eigen_energies:Dict[str, np.ndarray] = self.get_eigen_energies()
        
        ### Step 4. 找到 cbm 的能量、自旋、kpoint、能带
        cbm_energy = float("inf")   # energy
        cbm_kpt = 0     # kpoints 的索引 
        cbm_band = 0    # band 的索引
        cbm_spin = None # 自旋: Optional["up"|"down"]
        for tmp_spin in list( spin2eigen_energies.keys() ): # ["up", "down"]
            for idx_row, idx_col in zip( *np.where(spin2eigen_energies[tmp_spin] >= efermi_ev) ):
                # idx_row:index for kpoints ; idx_col: index for band
                if spin2eigen_energies[tmp_spin][idx_row][idx_col] < cbm_energy:
                    cbm_energy = round( float(spin2eigen_energies[tmp_spin][idx_row][idx_col]), 4)
                    cbm_kpt = idx_row + 1
                    cbm_band = idx_col + 1
                    cbm_spin = tmp_spin
                    
        ### Step 5. Get all other band sharing the cbm
        cbm_dict = {
                "energies": [cbm_energy],
                "kpts": [cbm_kpt],
                "bands": [cbm_band],
                "spins": [cbm_spin],
        }
        for tmp_spin in list( spin2eigen_energies.keys() ):
            for idx_band in range(spin2eigen_energies["up"].shape[1]):
                try:
                    if math.fabs( spin2eigen_energies[tmp_spin][cbm_kpt][idx_band] - cbm_energy) < 0.001:
                        cbm_dict["energies"].append(cbm_energy)
                        cbm_dict["kpts"].append(cbm_kpt)
                        cbm_dict["bands"].append(idx_band + 1)
                        cbm_dict["spins"].append(tmp_spin)
                except: # spin2eigen_energies["down"] 为空时，会触发 `IndexError`
                    pass
        
        return cbm_dict
        
        
    def get_vbm(self, out_fermi_path:str):
        '''
        Description
        -----------
            1. 得到半导体的vbm
        
        Return
        ------
            1. Union[Dict, None]
            2. vbm_dict: Dict
                - e.g. {'energies': [-1.9866], 'kpts': [29], 'bands': [52], 'spins': ['up']}
            3. 当体系是金属的时候返回 None
        '''
        ### Step 1. 判断体系是否是半导体
        if self._is_metal(out_fermi_path=out_fermi_path):
            #print("本材料体系是金属")
            #raise SystemExit
            return None

        ### Step 2. 得到体系的费米能级
        out_fermi_object = OutFermi(out_fermi_path=out_fermi_path)
        efermi_ev = out_fermi_object.get_efermi()
        
        ### Step 3. 得到本征能量
        spin2eigen_energies:Dict[str, np.ndarray] = self.get_eigen_energies()
        
        ### Step 4. 得到 vbm 的能量、自旋、kpoint、能带
        vbm_energy = -float("inf")
        vbm_kpt = 0
        vbm_band = 0
        vbm_spin = None
        for tmp_spin in list( spin2eigen_energies.keys() ):
            for idx_row, idx_col in zip( *np.where(spin2eigen_energies[tmp_spin] <= efermi_ev) ):
                # idx_row:indx of kpoints ; idx_col: index of band
                if spin2eigen_energies[tmp_spin][idx_row][idx_col] > vbm_energy:
                    vbm_energy = round( float(spin2eigen_energies[tmp_spin][idx_row][idx_col]), 4 )
                    vbm_kpt = idx_row + 1
                    vbm_band = idx_col + 1
                    vbm_spin = tmp_spin
        
        ### Step 5. Get all othr band sharing the vbm
        vbm_dict = {
            "energies": [vbm_energy],
            "kpts": [vbm_kpt],
            "bands": [vbm_band],
            "spins": [vbm_spin],
        }
        for tmp_spin in list( spin2eigen_energies.keys() ):
            for idx_band in range(spin2eigen_energies["up"].shape[1]):
                try:
                    if math.fabs( spin2eigen_energies[tmp_spin][vbm_kpt][idx_band] - vbm_energy) < 0.001:
                        vbm_dict["energies"].append(vbm_energy)
                        vbm_dict["spins"].append(tmp_spin)
                        vbm_dict["bands"].append(idx_band)
                        vbm_dict["kpts"].append(vbm_kpt)
                except: # spin2eigen_energies["down"] 为空时，会触发 `IndexError`
                    pass
        
        return vbm_dict
    
    
    def get_bandgap(self, out_fermi_path:str):
        '''
        Description
        -----------
            1. 得到 bandgap
        
        Paramter
        --------
            1. out_fermi_path: str
                - OUT.FERMI 的绝对路径
        
        Return
        ------
            1. bandgap: float
                - unit: eV
        '''
        ### Step 1. 得到 CBM 和 VBM
        ### (在 `self.get_cbm()`` 和 `self.get_vbm()` 中判断是金属 or 半导体)
        vbm_dict = self.get_vbm(out_fermi_path=out_fermi_path)
        cbm_dict = self.get_cbm(out_fermi_path=out_fermi_path)
        
        ### Step 2. 得到带隙的大小
        bandgap = cbm_dict["energies"][0] - vbm_dict["energies"][0]
        
        return bandgap
    
    
    def get_bandgap_type(self, out_fermi_path:str):
        '''
        Description
        -----------
            1. 得到带隙类型
        
        Return
        ------
            1. int:
                - 0: 间接带隙
                - 1: 直接带隙
        '''
        ### Step 1. 得到 CBM 和 VBM
        ### (在 `self.get_cbm()`` 和 `self.get_vbm()` 中判断是金属 or 半导体)
        vbm_dict = self.get_vbm(out_fermi_path=out_fermi_path)
        cbm_dict = self.get_cbm(out_fermi_path=out_fermi_path)
        
        ### Step 2. 得到带的类型
        intersection, idx_1_lst, idx_2_lst = np.intersect1d(
                                                vbm_dict["kpts"],
                                                cbm_dict["kpts"],
                                                return_indices=True,
                                                )
        
        if intersection.size == 0:
            return 0    # 间接带隙
        else:
            return 1    # 直接带隙