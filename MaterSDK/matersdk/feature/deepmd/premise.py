from abc import ABC, abstractclassmethod
import numpy as np

from ...io.publicLayer.neigh import StructureNeighborsV1


class DpFeaturePairPremiseDescriptor(object):
    '''
    Description
    -----------
        1. Map str to Derived class of `StructureNeighborBase`.
    
    Usage
    -----
        1. Demo 1.
            ```python
            dppd = DpFeaturePairPremiseDescription("v1")
            ```
    
    ----
        1. 'v1': `DpFeaturePairPremise`
            - for `StructureNeighborsV1`
    '''
    registry = {}

    @classmethod
    def register(cls, name:str):
        def wrapper(subclass:DpFeaturePairPremiseBase):
            cls.registry[name] = subclass
        return wrapper
    
    
    @classmethod
    def create(cls, name:str, *args, **kwargs):
        subclass = cls.registry[name]
        if subclass is None:
            raise ValueError(f"No DpFeaturePairPremise registered with name '{name}'")
        return subclass(*args, **kwargs)
    


class DpFeaturePairPremiseBase(ABC):
    @abstractclassmethod
    def extract_feature_pair(self):
        pass
    
    

@DpFeaturePairPremiseDescriptor.register("v1")
class DpFeaturePairPremiseV1(DpFeaturePairPremiseBase):
    def __init__(
                self,
                structure_neighbors:StructureNeighborsV1,
                ):
        self.structure_neighbors = structure_neighbors
        self.max_num_nbrs_real = structure_neighbors.key_nbr_atomic_numbers.shape[1]
    
    
    def extract_feature_pair(
                self,
                center_atomic_number:int,
                nbr_atomic_number:int):
        '''
        Description
        -----------
            1. 根据以下几个条件筛选所需要的信息：
                1.1. `中心原子的元素种类`
                1.2. `近邻原子的元素种类`
                1.3. `排除中心原子`
                1.4. `最大近邻原子数` (对于一种近邻元素来说)，决定了 zero_padding 的尺寸
        
        Parameters
        ----------
            1. center_atomic_number: int
                - 中心原子的原子序数
            2. nbr_atomic_number: int
                - 近邻原子的原子序数
        '''
        ### Step 1. 根据中心原子的原子序数，设置筛选条件 -- `mask_center`
        ### Step 1.1. 根据中心原子的原子序数的筛选条件
        mask_center = (self.structure_neighbors.key_nbr_atomic_numbers[:, 0] == center_atomic_number)
        mask_center = np.array(mask_center).reshape(-1, 1)
        mask_center = np.repeat(mask_center, self.max_num_nbrs_real, axis=1)
        ### Step 1.2. 根据近邻原子的原子序数的筛选条件
        mask_nbr = (self.structure_neighbors.key_nbr_atomic_numbers == nbr_atomic_number)
        ### Step 1.3. 获取上述筛选条件的 &
        mask_total = mask_center & mask_nbr
        ### Step 1.4. 排除中心原子自身
        # shape = (num_centers, max_num_nbrs_real)
        '''
        [[False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False False False False False False False]
        [False False False False False False False  True  True  True  True  True True]
        [False False False False False False False  True  True  True  True  True True]
        [False False False False False False False  True  True  True  True  True True]
        [False False False False False False False  True  True  True  True  True True]]
        '''
        mask_total[:, 0] = False
        max_num_nbrs_real = np.max(np.count_nonzero(mask_total, axis=1))
        
        ### Step 1.5. 有效行数
        mask_efficient_rows = np.where(
                np.count_nonzero(mask_center, axis=1),
                True,
                False
        )
        num_efficient_rows = np.count_nonzero(mask_efficient_rows)
        

        ### Step 2.
        setattr( self, "num_centers", num_efficient_rows )
        setattr( self, "max_num_nbrs_real_element", max_num_nbrs_real)
        
        ### Step 3. 初始化返回的数组 (这些数组都不包括中心原子自身)
        #       1. `dp_feature_pair_an`:
        #       2. `dp_feature_pair_d`:
        #       3. `dp_feature_pair_rc`:
        # shape = (num_efficient_rows, max_num_nbrs_reals) -- num_efficient_rows < num_centers
        dp_feature_pair_an = np.zeros((self.num_centers, self.max_num_nbrs_real_element))
        # shape = (num_efficient_rows, max_num_nbrs_reals) -- num_efficient_rows < num_centers
        dp_feature_pair_d = np.zeros((self.num_centers, self.max_num_nbrs_real_element))
        # shape = (num_efficient_rows, max_num_nbrs_reals, 3) -- num_efficient_rows < num_centers
        dp_feature_pair_rc = np.zeros((self.num_centers, self.max_num_nbrs_real_element, 3))

        ### Step 4. 计算相对坐标
        center_coords = self.structure_neighbors.key_nbr_coords[:, 0, :]
        center_coords = np.repeat(
                            center_coords[:, np.newaxis, :],
                            1,
                            axis=1)
        # shape = (num_centers, max_num_nbrs_max+1, 3) = (12, 13, 3)
        relative_coords = self.structure_neighbors.key_nbr_coords - center_coords
        
        ### Step 5. 根据筛选条件 `mask_total` 填充 Step 3 的三个数组
        efficient_center_idx = 0    # 记录填写到第几个有效行
        for tmp_center in range(relative_coords.shape[0]):
            tmp_mask = mask_total[tmp_center, :]
            if mask_efficient_rows[tmp_center]:
                tmp_max_num_nbrs_real = np.count_nonzero(tmp_mask)
                dp_feature_pair_an[efficient_center_idx, :tmp_max_num_nbrs_real] = self.structure_neighbors.key_nbr_atomic_numbers[tmp_center][tmp_mask]
                dp_feature_pair_d[efficient_center_idx, :tmp_max_num_nbrs_real] = self.structure_neighbors.key_nbr_distances[tmp_center][tmp_mask]
                dp_feature_pair_rc[efficient_center_idx, :tmp_max_num_nbrs_real, :] = relative_coords[tmp_center][tmp_mask][:]
                efficient_center_idx += 1
        assert (efficient_center_idx == num_efficient_rows)
        
        return dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc
    

    def expand_rc(
                self,
                center_atomic_number:int,
                nbr_atomic_number:int,
                max_num_nbrs:int):
        '''
        Description
        -----------
            1. This function is just aimed to fit Siyu's PWmatMLFF
            2. 根据`max_num_nbrs`, 扩展 `dp_feature_pair_rc` 
                - 需要先调用 `self.extract_feature_pair()` 获取 `dp_feature_pair_rc`
            
        Return
        ------
            1. expanded_rc: np.ndarray
                - .shape = (num_centers, max_num_nbrs, 4)
        '''        
        dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc = \
                self.extract_feature_pair(
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_number=nbr_atomic_number
                )
        del dp_feature_pair_an
        del dp_feature_pair_d
        assert(max_num_nbrs >= dp_feature_pair_rc.shape[1])
        
        num_centers = dp_feature_pair_rc.shape[0]
        max_num_nbrs_real = dp_feature_pair_rc.shape[1]
        expanded_rc  = np.zeros((num_centers, max_num_nbrs, 3))
        
        expanded_rc[:, :max_num_nbrs_real, :] = dp_feature_pair_rc
        
        return expanded_rc