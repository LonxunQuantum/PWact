import os
import numpy as np
from typing import List


from ...io.publicLayer.structure import DStructure
from ...io.pwmat.output.movement import Movement
from ...io.publicLayer.neigh import (
                                StructureNeighborsDescriptor,
                                StructureNeighborsV1
                                )
from ...feature.deepmd.premise import DpFeaturePairPremiseDescriptor


class AvgAngle(object):
    def __init__(self):
        pass
    
    
    @staticmethod
    def get_avg_bond(
            struct_neigh:StructureNeighborsV1,
            atomic_number_1:int, 
            atomic_number_2:int,
            atomic_number_3:int):
        '''
        Description
        -----------
            1. For xhm, useless function.
            2. 得到 `atomic_number_2 - atomic_number_1 - atomic_number_3` 的平均键角
    
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
        
        ### Step 2. 按照近邻原子种类筛选
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
        
        ### Step 3. 已知两个三维向量，计算其夹角 （单位：度）
        def get_angle(array_1:np.ndarray, array_2:np.ndarray):
            dot_product = np.dot(array_1, array_2)
            mag_1 = np.linalg.norm(array_1)
            mag_2 = np.linalg.norm(array_2)
            cos_value = dot_product / (mag_1 * mag_2)
            angle_radians = np.arccos(cos_value)
            angle_degree = np.degrees(angle_radians)    
            return angle_degree
        
        ### Step 4. 计算所有 `atomic_number_2 - atomic_number_1 - atomic_number_3` 的角度，并添加到 `angles_array`
        angles_array = []
        ### Step 4.1. Case 1.
        if (atomic_number_2 == atomic_number_3):
            for center_idx in range(key_nbr2_rc.shape[0]):
                for nbr2_idx in range(key_nbr2_rc.shape[1]):
                    for nbr3_idx in range(nbr2_idx, key_nbr3_rc.shape[1]):
                        tmp_nbr2_rc = key_nbr2_rc[center_idx, nbr2_idx, :]
                        tmp_nbr3_rc = key_nbr3_rc[center_idx, nbr3_idx, :]
                        # relative_coord 不能为 [0, 0, 0] ([0, 0, 0]代表无近邻原子)
                        if (not np.array_equal(tmp_nbr2_rc, np.zeros(3))) and \
                            (not np.array_equal(tmp_nbr3_rc, np.zeros(3))):
                                if (get_angle(tmp_nbr2_rc, tmp_nbr3_rc) >= 0):
                                    angles_array.append( get_angle(tmp_nbr2_rc, tmp_nbr3_rc) )      
                                        
        ### Step 4.2. Case 2. 
        else:
            for center_idx in range(key_nbr2_rc.shape[0]):
                for nbr2_idx in range(key_nbr2_rc.shape[1]):
                    for nbr3_idx in range(key_nbr3_rc.shape[1]):
                        tmp_nbr2_rc = key_nbr2_rc[center_idx, nbr2_idx, :]
                        tmp_nbr3_rc = key_nbr3_rc[center_idx, nbr3_idx, :]
                        # relative_coord 不能为 [0, 0, 0] ([0, 0, 0]代表无近邻原子)
                        if (not np.array_equal(tmp_nbr2_rc, np.zeros(3))) and \
                                (not np.array_equal(tmp_nbr3_rc, np.zeros(3))):
                            angles_array.append( get_angle(tmp_nbr2_rc, tmp_nbr3_rc) )
        
        
        ### Step 5. 
        angles_array = np.array(angles_array)
        result = 1 - 3/8 * np.sum(np.power(1/3 + np.cos(np.deg2rad(angles_array)), 2))
        
        return np.mean(angles_array), result