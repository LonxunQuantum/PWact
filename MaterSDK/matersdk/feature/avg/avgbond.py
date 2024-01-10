import os
import numpy as np
from typing import List
import multiprocessing as mp

from ...io.publicLayer.structure import DStructure
from ...io.pwmat.output.movement import Movement
from ...io.publicLayer.neigh import (
                                StructureNeighborsDescriptor,
                                StructureNeighborsV1
                                )
from ...feature.deepmd.premise import DpFeaturePairPremiseDescriptor
from ...io.pwmat.utils.parameters import specie2atomic_number



class AvgBond(object):
    def __init__(self,
                movement_path:str,
                element_1:str,
                element_2:str,
                rcut:float):
        '''
        Parameters
        ----------
            1. movement_path: str
                - MOVEMENT 的路径
            2. element_1: str
                - 成键的第一种元素
            3. element_2:
                - 成键的第二种元素
            4. rcut:
                - 成键的标准
        '''
        self.frames_lst = Movement(movement_path=movement_path).get_all_frame_structures_info()[0]
        self.atomic_number_1 = specie2atomic_number[element_1]
        self.atomic_number_2 = specie2atomic_number[element_2]
        self.rcut = rcut
        self.num_sites = self.frames_lst[0].num_sites
        
    
    def get_frames_avg_bond(
                    self,
                    scaling_matrix:List[int]=[3, 3, 3],
                    reformat_mark:bool=True,
                    coords_are_cartesian:bool=True):
        '''
        Description
        -----------
            1. 计算 Movement 中所有 frame 的平均键长
        
        Parameters
        ----------
            1. scaling_matrix: List[int]
                - bulk: [3, 3, 3]
                - slab: [3, 3, 1]
            2. reformat_mark: bool
                - 是否按原子序数从小到大排序
            3. coords_are_cartesian: bool
                - 是否按照笛卡尔坐标计算
        '''
        parameters_lst = []
        avg_bond_lengths_lst:List[float] = []
    
        parameters_lst = [(
                    tmp_structure,
                    self.rcut,
                    scaling_matrix,
                    self.atomic_number_1,
                    self.atomic_number_2) for tmp_structure in self.frames_lst]
        
        with mp.Pool(os.cpu_count()-2) as pool:
            avg_bond_lengths_lst = pool.starmap(
                                        ParallelFunction.get_avg_bond_length,
                                        parameters_lst
            )
        
        return avg_bond_lengths_lst
    
    
    @staticmethod
    def get_avg_bond_length(
                struct_neigh:StructureNeighborsV1,
                atomic_number_1:int,
                atomic_number_2:int):
        '''
        Description
        -----------
            1. 根据 `StructureNeighborVX` 的信息计算 `某一帧构型` 的 `atomic_number_1 - atomic_number_2`平均键长 (e.g. 所有 `Ge-Te` 对的长度)

        Parameters
        ----------
            1. struct_neigh: StructureNeighborsV#
                - 近邻原子信息
        '''
        key_nbr_ans = struct_neigh.key_nbr_atomic_numbers
        key_nbr_distances = struct_neigh.key_nbr_distances
        
        ### Step 1. 按照中心原子种类筛选
        filter_center = np.where(
                        key_nbr_ans[:, 0] == atomic_number_1,
                        True,
                        False)
        filter_center = np.repeat(
                        filter_center[:, np.newaxis],
                        key_nbr_ans.shape[1],
                        axis=1)
        
        ### Step 2. 按照近邻原子种类筛选
        filter_nbr = np.where(
                        key_nbr_ans == atomic_number_2,
                        True,
                        False)
     
        ### Step 3. 取 Step_1 和 Step_2 的and
        filter_tot = filter_center & filter_nbr
        ### Note: 中心原子全部取 False !!!
        filter_tot[:, 0] = False

        ### Step 4. 计算有效的键长，并存储为 np.array 格式
        effective_bonds_array = np.where(
                            filter_tot,
                            key_nbr_distances,
                            0)
        
        ### Step 5. 计算键长之和与键的数目
        num_bonds = np.count_nonzero(effective_bonds_array)
        sumlength_bonds = np.sum(effective_bonds_array)
        
        return sumlength_bonds / num_bonds


    @staticmethod
    def get_bond_lengths_lst_according2angle(
                struct_neigh:StructureNeighborsV1,
                atomic_number_1:int,
                atomic_number_2:int,
                atomic_number_3:int,
                angle_standard:float,
                angle_epsilon:float):
        '''
        Description
        -----------
            1. Just for xhm, useless function.
        
        Parameters
        ----------
            1. struct_neigh: StructureNeighborsV#
                - neighbor list
            2. atomic_number_1: int
                - 成键元素 1 (中心元素)
            3. atomic_number_2: int
                - 成键原子 2
            4. atomic_number_3: int
                - 成键原子 3
        
        Return
        ------
            1. 
        '''
        ### Step 1. 按照中心原子种类筛选
        filter_center = np.where(
                struct_neigh.key_nbr_atomic_numbers[:, 0] == atomic_number_1,
                True,
                False)
        filter_center = np.repeat(
                filter_center[:, np.newaxis],
                struct_neigh.key_nbr_atomic_numbers.shape[1],
                axis=1)

        ### Step 2. 按照 `近邻原子` 种类筛选
        dp_feature = DpFeaturePairPremiseDescriptor.create(
                            "v1",
                            structure_neighbors=struct_neigh)
        
        ### Step 2.2. 按照 `近邻原子_2` 种类筛选 -- `key_nbr2_rc`.shape = (num_center, num_nbrs, 3)
        key_nbr2_an, key_nbr2_d, key_nbr2_rc = \
            dp_feature.extract_feature_pair(
                        center_atomic_number=atomic_number_1,
                        nbr_atomic_number=atomic_number_2)
        
        ### Step 2.3. 按照 `近邻原子_3` 种类筛选 -- `key_nbr3_rc`
        key_nbr3_an, key_nbr3_d, key_nbr3_rc = \
            dp_feature.extract_feature_pair(
                        center_atomic_number=atomic_number_1,
                        nbr_atomic_number=atomic_number_3) 
        
        ### Step 3. 已知两个三维向量，计算其夹角 (单位：度)
        def get_angle(array_1:np.ndarray, array_2:np.ndarray):
            dot_product = np.dot(array_1, array_2)
            mag_1 = np.linalg.norm(array_1)
            mag_2 = np.linalg.norm(array_2)
            cos_value = dot_product / (mag_1 * mag_2)
            angle_radians = np.arccos(cos_value)
            angle_degree = np.degrees(angle_radians)
            return angle_degree
        
        
        ### Step 4. angle 是否在 误差范围内
        def get_angle_equal(
                    angle:float,
                    angle_standard:float,
                    angle_epsilon:float):
            abs_deviation = abs(angle - angle_standard)
            if (abs_deviation < angle_epsilon):
                return True
            else:
                return False
        
        
        ### Step 5. 
        shorter_bonds_array = []
        longer_bonds_array = []
        angles_array = []
        ### Step 5.1. Case 1 -- 
        if (atomic_number_2 == atomic_number_3):
            for center_idx in range(key_nbr2_rc.shape[0]):
                for nbr2_idx in range(key_nbr2_rc.shape[1]):
                    for nbr3_idx in range(nbr2_idx, key_nbr3_rc.shape[1]):  # Note: 预防重复计算键角
                        tmp_nbr2_rc = key_nbr2_rc[center_idx, nbr2_idx, :]
                        tmp_nbr3_rc = key_nbr3_rc[center_idx, nbr3_idx, :]
                        # relative_coord 不能为 [0, 0, 0] ([0, 0, 0]代表无近邻原子)
                        if (not np.array_equal(tmp_nbr2_rc, np.zeros(3))) and \
                            (not np.array_equal(tmp_nbr3_rc, np.zeros(3))):
                                # 角度满足 angle_standard ± angle_epsilon
                                if ( get_angle_equal(
                                        angle=get_angle(tmp_nbr2_rc, tmp_nbr3_rc),
                                        angle_standard=angle_standard,
                                        angle_epsilon=angle_epsilon
                                ) ):
                                    angles_array.append(get_angle(tmp_nbr2_rc, tmp_nbr3_rc))
                                    shorter_bonds_array.append( min([np.linalg.norm(tmp_nbr2_rc), np.linalg.norm(tmp_nbr3_rc)]) )
                                    longer_bonds_array.append( max([np.linalg.norm(tmp_nbr2_rc), np.linalg.norm(tmp_nbr3_rc)]) )
                                    
            
            
        ### Step 5.2. Case 2 -- 
        else:
            for center_idx in range(key_nbr2_rc.shape[0]):
                for nbr2_idx in range(key_nbr2_rc.shape[1]):
                    for nbr3_idx in range(key_nbr3_rc.shape[1]):
                        tmp_nbr2_rc = key_nbr2_rc[center_idx, nbr2_idx, :]
                        tmp_nbr3_rc = key_nbr3_rc[center_idx, nbr3_idx, :]
                        # relative_coord 不能为 [0, 0, 0] ([0, 0, 0]代表无近邻原子)
                        if (not np.array_equal(tmp_nbr2_rc, np.zeros(3))) and \
                                (not np.array_equal(tmp_nbr3_rc, np.zeros(3))):
                            # 角度满足 angle_standard ± angle_epsilon
                            if( get_angle_equal(
                                    angle=get_angle(tmp_nbr2_rc, tmp_nbr3_rc),
                                    angle_standard=angle_standard,
                                    angle_epsilon=angle_epsilon) ):
                                angles_array.append(get_angle(tmp_nbr2_rc, tmp_nbr3_rc))
                                shorter_bonds_array.append( min([np.linalg.norm(tmp_nbr2_rc), np.linalg.norm(tmp_nbr3_rc)]) )
                                longer_bonds_array.append( max([np.linalg.norm(tmp_nbr2_rc), np.linalg.norm(tmp_nbr3_rc)]) )
        
        ### Step 6. 合并
        angles_array = np.array( angles_array ).reshape(-1, 1)
        shorter_bonds_array = np.array( shorter_bonds_array ).reshape(-1, 1)
        longer_bonds_array = np.array( longer_bonds_array ).reshape(-1, 1)
                            
        result_array = np.concatenate((angles_array, shorter_bonds_array, longer_bonds_array), axis=1)
        
        return result_array
        
        
        


class ParallelFunction(object):
    @staticmethod
    def get_avg_bond_length(
                    structure:DStructure,
                    rcut:float,
                    scaling_matrix:List[int],
                    atomic_number_1:int,
                    atomic_numner_2:int):
        struct_neigh = StructureNeighborsDescriptor.create(
                    'v1',
                    structure,
                    rcut,
                    scaling_matrix)
        avg_bond_length = AvgBond.get_avg_bond_length(
                                struct_neigh=struct_neigh,
                                atomic_number_1=atomic_number_1,
                                atomic_number_2=atomic_numner_2)
        return avg_bond_length




class PairBond(object):
    def __init__(
                self,
                movement_path:str,
                atom1_idx:int,
                atom2_idx:int,
                ):
        '''
        Description
        -----------
            1. 
        
        Parameters
        ----------
            1. movement_path: str
            2. atom1_idx: int 
            3. atom2_idx: int
        '''
        self.frames_lst = Movement(movement_path=movement_path).get_all_frame_structures_info()[0]
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx
    
    
    def get_frames_pair_bond(self):
        pair_bond_lengths_lst = []
        for tmp_struct in self.frames_lst:
            tmp_coords = tmp_struct.cart_coords
            tmp_atom1_coord = tmp_coords[self.atom1_idx]
            tmp_atom2_coord = tmp_coords[self.atom2_idx]
            tmp_pair_bond = self._get_pair_bond_length(tmp_atom1_coord, tmp_atom2_coord)
            pair_bond_lengths_lst.append(tmp_pair_bond)
        
        return pair_bond_lengths_lst
            
    
    
    def _get_pair_bond_length(
                            self,
                            atom1_coord:np.ndarray,
                            atom2_coord:np.ndarray
                            ):
        '''
        Description
        -----------
            1. 计算两个坐标之间的距离
        '''
        return np.linalg.norm(atom2_coord - atom1_coord)