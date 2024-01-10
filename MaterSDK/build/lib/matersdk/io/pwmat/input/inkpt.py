import linecache
import numpy as np
from typing import List

from ...publicLayer.structure import DStructure


class Inkpt(object):
    '''
    Description
    -----------
        1. This class is aimed to `IN.KPT`
    
    Note
    ----
        1. 注意单位换算：埃 <-> Bohr
            - IN.KPT 中用的是 `埃`
            - REPORT 中用的是 `Bohr`
        2. 目前仅支持 `iflag, a0 = 2, 0`
    '''
    def __init__(self, in_kpt_path:str):
        '''
        Parameters
        ----------
            1. in_kpt_path: str
                - IN.KPT 文件的路径
        '''
        self.in_kpt_path = in_kpt_path
        self.iflag = self._get_iflag()
        self.a0 = self._get_a0()
    
    
    def _get_iflag(self):
        '''
        Description
        -----------
            1. 得到 IN.KPT 中的 `iflag`
                - 用于判断 `K 点位置定义在 x, y, z 方向` 或 `K 点位置定义在 AL(3,3) 的倒格子中`
        
        Return
        ------
            1. a0: int
                - 
        '''
        line_2 = linecache.getline(self.in_kpt_path, 2)
        iflag = int( line_2.split()[0] )
        return iflag
    
    
    def _get_a0(self):
        '''
        Description
        -----------
            1. 得到 IN.KPT 中的 `a0`
                - 仅在 iflag = 1 时使用 (原子单位 Bohr)
        
        Return
        ------
            1. a0: int
                - 
        '''
        line_2 = linecache.getline(self.in_kpt_path, 2)
        a0 = int( line_2.split()[-1] )
        return a0
    
    
    def get_num_kpts(self):
        '''
        Description
        -----------
            1. 得到 kpoints 的个数
        
        Return
        ------
            1. num_kpts: int
        '''
        line_1 = linecache.getline(self.in_kpt_path, 1)
        num_kpts = int( line_1.split()[0] )
        return num_kpts
    
    
    def get_kpt_coords_frac(self):
        '''
        Description
        -----------
            1. 得到 IN.KPT 中所有 KPOINTS 的分数坐标
            
        Return
        ------
            1. kpt_coord_frac: np.ndarray
                - 所有 kpoints 的坐标
        '''
        ### Step 1. 读取 `仅包含坐标和权重的行`
        with open(self.in_kpt_path, "r") as f:
            lines_lst = f.readlines()
        lines_lst = lines_lst[2:]
        
        ### Step 2. 取出所有的 Kpoints 坐标
        coords_frac_lst = []
        for tmp_line in lines_lst:
            tmp_coord_frac = [float(tmp_value) \
                        for tmp_value in tmp_line.split()[:3]]
            coords_frac_lst.append(tmp_coord_frac)
        coords_frac_array = np.array(coords_frac_lst)
        
        return coords_frac_array
    

    def get_kpt_coords_A(
                    self, 
                    atom_config_path:str):
        '''
        Description
        -----------
            1. 得到所有 kpoints 的坐标 (单位：埃)，与 IN.KPT/OUT.KPT 对应
        '''
        ### Step 1. 得到倒易晶格（unit: 埃）
        structure = DStructure.from_file(
                            file_path=atom_config_path,
                            file_format="pwmat",
                            )
        reciprocal_lattice_array = np.array( structure.lattice.reciprocal_lattice.matrix )

        ### Step 2. 得到kpoints的分数坐标
        kpt_coords_frac = self.get_kpt_coords_frac()
        
        ### Step 3. kpoints的分数坐标 * 倒易晶格
        return np.dot(kpt_coords_frac, reciprocal_lattice_array)        
    
    
    def get_kpt_coords_Bohr(
                    self, 
                    atom_config_path:str):
        '''
        Description
        -----------
            1. 得到所有 kpoints 的坐标 (单位：Bohr)，与 REPORT 对应
        '''
        ### Step 1. 得到倒易晶格（unit: 埃）
        structure = DStructure.from_file(
                            file_path=atom_config_path,
                            file_format="pwmat",
                            )
        reciprocal_lattice_array = np.array( structure.lattice.reciprocal_lattice.matrix )

        ### Step 2. 得到kpoints的分数坐标
        kpt_coords_frac = self.get_kpt_coords_frac()
        
        ### Step 3. kpoints的分数坐标 * 倒易晶格
        return np.dot(kpt_coords_frac, reciprocal_lattice_array) * 0.529177249
    
    
    def get_kpt_weights(self):
        '''
        Description
        -----------
            1. 得到 kpoints 的权重
            
        Return
        ------
            1. weights_lst: List[float]
                - kpoints 的权重
        '''
        with open(self.in_kpt_path, "r") as f:
            lines_lst = f.readlines()[2:]
        
        weights_lst = [float(tmp_line.split()[3]) for tmp_line in lines_lst]
        
        return weights_lst
    
    
    def get_hsp(self):
        '''
        Description
        -----------
            1. 得到 IN.KPT 中的高对称点和分数坐标
        
        Return
        ------
            1. {
                'hsp': ['G', 'M', 'K', 'G', 'A', 'L', 'H', 'A', 'L', 'M', 'K', 'H'], 
                'coords_frac': [
                        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.333333, 0.333333, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.333333, 0.333333, 0.5], [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 0.0], [0.333333, 0.333333, 0.0], [0.333333, 0.333333, 0.5]
                        ]
                }
        '''
        hsp_coords_dict = {"hsp":[], "coords_frac":[]}
        with open(self.in_kpt_path, "r") as f:
            lines_lst = f.readlines()[2:]
        
        for tmp_line in lines_lst:
            if ( len(tmp_line.split()) == 5):
                tmp_line_lst = tmp_line.split()
                hsp_coords_dict["hsp"].append(tmp_line_lst[-1])
                hsp_coords_dict["coords_frac"].append(
                        [float(tmp_line_lst[0]), float(tmp_line_lst[1]), float(tmp_line_lst[2])]
                        )
        
        return hsp_coords_dict
    
    
    def _get_idx2hsp(self):
        '''
        Description
        -----------
            1. 得到所有高对称点的 index (从 0 开始)
        
        Return
        ------
            1. idx2hsp: Dict[int, str]:
                - Dict[高对称点的index, 高对称点]
        '''
        idx2hsp = {}
        with open(self.in_kpt_path) as f:
            lines_lst = f.readlines()[2:]
        
        for idx_line, tmp_line in enumerate(lines_lst):
            if ( len(tmp_line.split()) == 5 ):
                tmp_line_lst = tmp_line.split()
                idx2hsp.update(
                        {idx_line:tmp_line_lst[-1]}
                        )
        
        return idx2hsp

    
    def _get_distance_from_nbr(
                            self,
                            atom_config_path:str):
        '''
        Description
        -----------
            1. 得到 index近邻(在IN.KPT中) 的两个 kpoints 的距离 -- 单位：埃
        
        Return
        ------
            1. distances_from_nbr_lst: List[float]
                - index近邻(在IN.KPT中) 的两个 kpoints 的距离 -- 单位：埃
        '''
        kpt_coords_A = self.get_kpt_coords_A(
                        atom_config_path=atom_config_path)
        distances_from_nbr_lst = \
                list( 
                    np.linalg.norm( 
                    np.diff(kpt_coords_A, axis=0),
                    axis=1)
                    )
        distances_from_nbr_lst.insert(0, 0)
        
        assert (kpt_coords_A.shape[0] == len(distances_from_nbr_lst))
        
        return distances_from_nbr_lst
    
    
    def _split_distances_from_gamma_lst(
                            self,
                            atom_config_path:str,
                            ):
        '''
        Description
        -----------
            1. 将不同kpath上kpoints距gamma点的距离分成不同的列表 -- 单位：埃
            
        Return
        ------
            1. distance_from_gamma_in_kpaths_lst: List[ List[float] ]:
                - 
        '''
        ### distance_from_gamma_in_kpath_lst: 二维列表
        ###     1d: 不同的kpath 
        ###     2d: 某一kpath上所有kpoints距离gamma点的距离
        distance_from_gamma_in_kpaths_lst = []
        
        ### Step 1. 
        idx2hsp = self._get_idx2hsp()
        distances_from_nbr_lst = self._get_distance_from_nbr(atom_config_path=atom_config_path)
        distances_from_gamma_lst = list(
                np.cumsum(np.array(distances_from_nbr_lst))
        )
        
        ### Step 2. 得到每条 kpath 的第一个kpoint的索引 -- kpt_idx_starts_lst
        kpt_idx_starts_lst = [0]
        for tmp_idx in range( 1, len(idx2hsp.keys()) ):
            if ( (list(idx2hsp.keys())[tmp_idx] - list(idx2hsp.keys())[tmp_idx-1]) == 1):
                kpt_idx_starts_lst.append(list(idx2hsp.keys())[tmp_idx])
        kpt_idx_starts_lst.append( len(distances_from_nbr_lst) )
        #print(kpt_idx_starts_lst)
        
        ### Step 3. 将不同kpath上kpoint之间的距离，分成不同的list
        for tmp_idx in range(1, len(kpt_idx_starts_lst)):
            tmp_distances = distances_from_gamma_lst[kpt_idx_starts_lst[tmp_idx-1]:kpt_idx_starts_lst[tmp_idx]]
            distance_from_gamma_in_kpaths_lst.append(tmp_distances)
        
        ### Step 4. Test
        #for tmp_lst in distance_from_nbr_in_kpaths_lst:
        #    print(len(tmp_lst))
        
        return distance_from_gamma_in_kpaths_lst
            
    
    
    def get_distance_from_gamma_A(
                            self,
                            atom_config_path:str):
        '''
        Description
        -----------
            1. 得到二维能带图的横坐标 (unit: 埃)
        
        Return 
        ------
            1. distances_from_gamma_lst: List[float] -- 单位：埃 
                - 所有 kpoints 距 gamma点的距离
        
        Note
        ----
            1. 注意三维体系，具有不同的KPATH，此时存在`跳点问题`
        '''
        distances_from_gamma_lst:List[float] = []
        
        ### Step 1. 未处理跳点问题的 distances_from_gamma_lst_: List[ List[float] ]
        distances_from_gamma_lst_:List[float] = \
                            self._split_distances_from_gamma_lst(atom_config_path=atom_config_path)
        #print(distances_from_gamma_lst_)
        
        ### Step 2. 处理不同kpath的跳点问题
        for tmp_idx, tmp_distances_from_gamma in enumerate(distances_from_gamma_lst_): # 有多少 kpath，次循环就进行多少次！
            if (tmp_idx != 0):  # 如果是第一条 kpath
                cum_distance = distances_from_gamma_lst[-1]
                minus_distance = tmp_distances_from_gamma[0] 
                #print(cum_distance, minus_distance)
                new_tmp_distances_from_gamma = \
                        [distance+cum_distance-minus_distance for distance in tmp_distances_from_gamma]
            else:   # 如果不是第一条 kpath
                new_tmp_distances_from_gamma = tmp_distances_from_gamma
            
            for tmp_distance in new_tmp_distances_from_gamma:
                distances_from_gamma_lst.append(tmp_distance)
        
        return distances_from_gamma_lst
                    
    
    def get_distance_from_gamma_bohr(
                            self,
                            atom_config_path:str,    
                            ):
        '''
        Description
        -----------
            1. 得到二维能带图的横坐标 (unit: Bohr)
        
        Return 
        ------
            1. distances_from_gamma_lst: List[float] -- 单位：埃 
                - 所有 kpoints 距 gamma点的距离
        
        Note
        ----
            1. 注意三维体系，具有不同的KPATH，此时存在`跳点问题`
        '''
        BOHR = 0.529177249  # 埃
        distances_from_gamma_A_lst = \
                self.get_distance_from_gamma_A(
                        atom_config_path=atom_config_path)
        
        distances_from_gamma_bohr_lst = \
                [(value * BOHR) for value in distances_from_gamma_A_lst]
        
        return distances_from_gamma_bohr_lst