import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod

from .structure import DStructure
from ..pwmat.utils.parameters import atomic_number2specie, specie2atomic_number


class StructureNeighborsUtils(object):
    @staticmethod
    def get_max_num_nbrs_real(
                    structure:DStructure,
                    rcut:float,
                    scaling_matrix:List[int],
                    coords_are_cartesian:bool=True):
        '''
        Description
        -----------
            1. 获取最大近邻原子数目
        '''
        ### Step 0. 获取 primitive_cell 中原子在 supercell 中的 index
        key_idxs = structure.get_key_idxs(scaling_matrix=scaling_matrix)
        supercell = structure.make_supercell_(
                                scaling_matrix=scaling_matrix,
                                reformat_mark=True)
        
        ### Step 1. 获取supercell所有原子的(笛卡尔)坐标 -- `supercell_coords`
        if coords_are_cartesian:
            supercell_coords = supercell.cart_coords
        else:
            supercell_coords = supercell.frac_coords
        
        
        ### Step 2. 计算在截断半径内的最大原子数
        max_num_nbrs = 0
        for tmp_i, tmp_center_idx in enumerate(key_idxs):
            ### Step 2.1. 计算该中心原子与近邻原子的距离
            # shape = (3,) -> (1, 3)
            tmp_center_coord = supercell_coords[tmp_center_idx].reshape(1, 3)
            # shape = (num_supercell, 3)
            tmp_relative_coords = supercell_coords - tmp_center_coord
            # shape = (num_supercell,)
            tmp_distances = np.linalg.norm(tmp_relative_coords, axis=1)
            
            ### Step 2.2. 判断哪些近邻原子在截断半径内
            tmp_mark_rcut = np.where(tmp_distances<rcut, True, False)
            
            ### Step 2.3. 计算在截断半径内的原子数目
            tmp_num_nbrs = np.count_nonzero(tmp_mark_rcut)
            
            ### Step 2.4. 判断
            if tmp_num_nbrs > max_num_nbrs:
                max_num_nbrs = tmp_num_nbrs
        
        return max_num_nbrs - 1
    
    
    @staticmethod
    def get_max_num_nbrs_real_element(
                            structure:DStructure,
                            rcut:float,
                            nbr_elements:List[str],
                            scaling_matrix:List[int],
                            coords_are_cartesian:bool=True):
        '''
        Description
        -----------
            1. 获取某元素最大近邻原子数目 
                - {"O": 38, "H", 78}: 近邻的O的数目最多为38；近邻的H的数目最多为78
        '''
        ### Step 0. 获取 primitive_cell 中原子在 supercell 中的index
        key_idxs = structure.get_key_idxs(scaling_matrix=scaling_matrix)
        supercell = structure.make_supercell_(
                                    scaling_matrix=scaling_matrix,
                                    reformat_mark=True)
        
        ### Step 1. 获取supercell 中所有原子的(笛卡尔)坐标 -- `supercell_coords`
        if coords_are_cartesian:
            supercell_coords = supercell.cart_coords
        else:
            supercell_coords = supercell.frac_coords
        supercell_species = np.array([tmp_specie.Z for tmp_specie in supercell.species])
        
        ### Step 2. 计算在截断半径内的各元素最大原子数
        nbr_element2max_num = dict.fromkeys(nbr_elements, 0)
        for tmp_i, tmp_center_idx in enumerate(key_idxs):
            ### Step 2.1. 计算该中心原子与近邻原子的距离
            # shape = (3,) -> (1,3)
            tmp_center_coord = supercell_coords[tmp_center_idx].reshape(1, 3)
            # shape = (num_supercell, 3)
            tmp_relative_coords = supercell_coords - tmp_center_coord
            # shape = (num_supercell,)
            tmp_distances = np.linalg.norm(tmp_relative_coords, axis=1)
            
            ### Step 2.2. 判断哪些近邻原子在截断半径内
            tmp_mark_rcut = np.where(tmp_distances<rcut, True, False)
            
            ### Step 2.3. 
            for tmp_element in nbr_elements:
                tmp_atomic_number = specie2atomic_number[tmp_element]
                tmp_mark_element = np.where(supercell_species==tmp_atomic_number, True, False)
                tmp_mark_tot = tmp_mark_element & tmp_mark_rcut
                tmp_mark_tot[tmp_center_idx] = False  # 排除中心原子自身
                tmp_num_element = np.count_nonzero(tmp_mark_tot)
                if tmp_num_element > nbr_element2max_num[tmp_element]:
                    nbr_element2max_num.update({tmp_element: tmp_num_element})
        
        return nbr_element2max_num
    
    
    @staticmethod
    def _get_nbrs_indices(
                struct_nbr,
                center_atomic_number:int,
                nbr_atomic_number:int,
                max_num_nbrs:Union[bool, int]=False):
        '''
        Description
        -----------
            1. 此函数是为了适配 PWmat-MLFF 中的 neigh_list!
                - Note: 由于 Fortran 是从 1 开始的，
                    1. 因此 `return tmp_idx + 1`
                    2. `0` 代表没有原子
        
        Parameters
        ----------
            1. struct_nbr: `StructureNeighborsV1`
                1.1. key_nbr_atomic_numbers: np.ndarray
                    - shape = (num_centers, num_nbrs)
                1.2. key_nbr_coords: np.ndarray
                    - shape = (num_centers, num_nbrs)
            2. center_atomic_number: int
                - 中心原子的原子序数
            3. nbr_atomic_number: int
                - 近邻原子的原子序数
            4. max_num_nbrs: int
                - 用于 zero-padding
        '''
        structure = struct_nbr.structure
        key_nbr_atomic_numbers = struct_nbr.key_nbr_atomic_numbers
        key_nbr_coords = struct_nbr.key_nbr_coords
        
        ### Step 1. Get `new_nbr_atomic_numbers`, `new_nbr_coords`
        ### Step 1.1. 根据中心原子种类，设定 mask_center
        mask_center = np.where(
                    key_nbr_atomic_numbers[:, 0]==center_atomic_number,
                    True,
                    False)
        mask_center = np.repeat(
                    mask_center[:, np.newaxis], 
                    key_nbr_atomic_numbers.shape[1],
                    axis=1
        )
        
        ### Step 1.2. 根据近邻原子种类，设定 mask_nbr
        mask_nbr = np.where(
                    key_nbr_atomic_numbers==nbr_atomic_number,
                    True,
                    False)
        
        ### Step 1.3. `mask_center & mask_nbr`
        mask_tot = mask_center & mask_nbr
        
        ### Step 1.4. Remove center atom self
        mask_tot[:, 0] = False
        max_num_nbrs_real = np.max(np.count_nonzero(mask_tot, axis=1))
        
        ### Step 1.5. 根据 `mask_center & mask_nbr` 取出
        #          - key_nbr_coords
        ### Step 1.5.1. 确定 center_atomic_number 的行数
        # e.g. [ True  True  True  True  True  True  True  True False False False False]
        mask_efficient_rows = np.where(
                                np.count_nonzero(mask_center, axis=1),
                                True,
                                False
        )
        num_efficient_rows = np.count_nonzero(mask_efficient_rows)
        # shape = (num_efficient_rows, max_num_nbrs_reals, 3) -- num_efficient_rows < num_centers
        new_nbr_coords = np.zeros(
                            (num_efficient_rows, 
                            max_num_nbrs_real,
                            3)
        )
        
        efficient_center_idx = 0    # 记录填写到第几个有效行
        for tmp_center in range(key_nbr_coords.shape[0]):
            tmp_mask = mask_tot[tmp_center, :]
            if mask_efficient_rows[tmp_center]:
                tmp_max_num_nbrs_real = np.count_nonzero(tmp_mask)
                new_nbr_coords[efficient_center_idx, :tmp_max_num_nbrs_real, :] = key_nbr_coords[tmp_center][tmp_mask][:]
                efficient_center_idx += 1
        assert (efficient_center_idx == num_efficient_rows)
       
        ### Step 2. 获取 neigh_list for PWmatMLFF
        neigh_list = []
        for tmp_center_idx, tmp_center_nbr_coord in enumerate(new_nbr_coords):
            tmp_neigh_list = []
            for tmp_nbr_idx, tmp_nbr_coord in enumerate(tmp_center_nbr_coord):
                tmp_neigh_list.append(structure.get_site_index(site_coord=tmp_nbr_coord))
            neigh_list.append(tmp_neigh_list)
        neigh_list = np.array(neigh_list)
        
        ### Step 3. Zero-padding
        if max_num_nbrs:
            neigh_list_zero_padding = np.zeros((neigh_list.shape[0], max_num_nbrs))
            for tmp_idx in range(neigh_list.shape[0]):
                neigh_list_zero_padding[tmp_idx, :max_num_nbrs_real] = neigh_list[tmp_idx]
            return neigh_list_zero_padding
        
        return neigh_list


    @staticmethod
    def get_nbrs_indices(
                struct_nbr,
                center_atomic_numbers:List[int],
                nbr_atomic_numbers:List[int],
                max_num_nbrs:Union[bool, List[int]]=False):
        '''
        Description
        -----------
            1. 合并 pair 的 neigh_list:
                1. Li:
                    1) Li-Li : (48, 100)
                    2) Li-Si : (48, 80)
                2. Si
                    1) Si-Li: (24, 100)
                    2) Si-Si: (24, 80)
            2. 合并后：
                1. 
                    1) (72, 180)
        
        Return 
        ------
            1. neigh_list: np.ndarray
                - .shape = (72, 180)
        '''
        neigh_lists_lst = []
        for tmp_center_an in center_atomic_numbers:
            tmp_center_neigh_lists_lst = []
            for tmp_nbr_an_idx, tmp_nbr_an in enumerate(nbr_atomic_numbers):
                tmp_nbr_neigh_list:np.ndarray = \
                        StructureNeighborsUtils._get_nbrs_indices(
                            struct_nbr=struct_nbr,
                            center_atomic_number=tmp_center_an,
                            nbr_atomic_number=tmp_nbr_an,
                            max_num_nbrs=max_num_nbrs[tmp_nbr_an_idx],
                        )
                tmp_center_neigh_lists_lst.append(tmp_nbr_neigh_list)
            tmp_center_neigh_lists_array = np.concatenate(tmp_center_neigh_lists_lst, axis=1)
            neigh_lists_lst.append(tmp_center_neigh_lists_array)
        
        neigh_list = np.concatenate(neigh_lists_lst, axis=0)
        
        return neigh_list
                


class StructureNeighborsDescriptor(object):
    '''
    Description
    -----------
        1. Map str to Drived class of `StructureNeighborBase`.
    
    Usage
    -----
        1. Demo 1
            ```python
            snd_v1 = StructureNeighborsDescriptor.create("v1")
            ```
    
    -----
        1. 'v1': `StructureNeighborsV1`
    '''
    registry = {}
    
    @classmethod
    def register(cls, name:str):
        def wrapper(subclass:StructureNeighborsBase):
            cls.registry[name] = subclass
        return wrapper
    
    @classmethod
    def create(cls, name:str, *args, **kwargs):
        subclass = cls.registry[name]
        if subclass is None:
            raise ValueError(f"No StructureNeighbors registered with name '{name}'")
        return subclass(*args, **kwargs)



class StructureNeighborsBase(ABC):
    @abstractmethod
    def _get_key_neighs_info(self):
        pass
    
    
    def _get_max_num_nbrs(self):
        '''
        v1
        '''
        pass
    


@StructureNeighborsDescriptor.register("v1")
class StructureNeighborsV1(StructureNeighborsBase):
    '''
    Description
    -----------
        1. Set `rcut`, not `n_neighbors`, not `max_num_nbrs`.
        2. Save images(frames) in different folders, and their neighbors' size are different.
    '''
    def __init__(
                self,
                structure:DStructure,
                rcut:float=6.5,
                scaling_matrix:List[int]=[3,3,3],
                reformat_mark:bool=True,
                coords_are_cartesian:bool=True):
        '''
        Parameters
        ----------
            1. structure: DStructure
                - 结构
            2. rcut: float:
                - 截断半径
            3. scaling_matrix: List[int]
                - 扩胞系数
            4. reformat_mark: bool
                - 是否按照原子序数排序。
                - 这个参数一定要设置为 `True`
            5. coords_are_cartesian: bool
                - 是否使用笛卡尔坐标
        '''
        ### Step 1. 
        self.structure = structure
        self.supercell = self.structure.make_supercell_(
                                scaling_matrix=scaling_matrix,
                                reformat_mark=reformat_mark)

        ### Step 2.
        self.key_nbr_atomic_numbers, self.key_nbr_distances, self.key_nbr_coords = \
                self._get_key_neighs_info(
                        scaling_matrix=scaling_matrix,
                        rcut=rcut,
                        coords_are_cartesian=coords_are_cartesian)
    
    
    def _get_key_neighs_info(
                self,
                scaling_matrix:List[int],
                rcut:float,
                coords_are_cartesian:bool):
        '''
        Description
        -----------
            1. 
        
        Parameters
        ----------
            1. scaling_matrix: List[int]
                - 
            2. rcut: float 截断半径
                - 
            3. coords_are_cartesian: bool
                - 
        
        Return
        ------
            1. nbr_atomic_numbers: np.ndarray, shape = (num_center, n_neighbors)
                - 
            2. nbr_distances: np.ndarray, shape = (num_center, n_neighbors)
                - 
            3. nbr_coords: np.ndarray, shape = (num_center, n_neighbors, 3)
                - 
        '''
        ### Step 0. 获取 primitive_cell 中的原子在 supercell 中的 index
        key_idxs = self.structure.get_key_idxs(scaling_matrix=scaling_matrix)
        
        ### Step 1. 获取 supercell 的各种信息 (sites的原子序数、坐标)，便于后边直接从其中抽取信息
        ###         1) supercell_atomic_numbers     2) supercell_coords
        ### Step 1.1. 获取 supercell 的各位点的原子序数 -- `supercell_atomic_numbers`
        supercell_atomic_numbers = np.array([tmp_site.specie.Z for tmp_site in self.supercell.sites])
        
        ### Step 1.2. 获取 supercell 的 (笛卡尔) 坐标 -- `supercell_coords`
        if coords_are_cartesian:
            supercell_coords = self.supercell.cart_coords
        else:
            supercell_coords = self.supercell.frac_coords
        
        ### Step 2. 初始化需要返回的三个 np.ndarray
        #   nbr_atomic_numbers: 近邻原子的元素种类 (原子序数)
        #   nbr_distances: 近邻原子距中心原子的距离
        #   nbr_coords: 近邻原子的坐标
        # shape = (num_center, n_neighbors)
        nbr_atomic_numbers = np.zeros((len(key_idxs), supercell_atomic_numbers.shape[0]))
        # shape = (num_center, n_neighbors)
        nbr_distances = np.zeros((len(key_idxs), supercell_atomic_numbers.shape[0]))
        # shape = (num_center, n_neighbors, 3)
        nbr_coords = np.zeros((len(key_idxs), supercell_atomic_numbers.shape[0], 3))
        
        
        ### Step 2.1. 每个 primitiv_cell 中的原子，循环一次
        max_num_nbrs = 0
        for tmp_i, tmp_center_idx in enumerate(key_idxs):
            '''
            Note
            ----
                1. `tmp_i`: 从 0 开始
                2. `tmp_center_idx`: primitive_cell 的原子在 supercell 中的 index
                3. `tmp_nbr_idxs`: 将 `supercell 中所有原子的索引`按照距中心原子距离的远近排序
            '''
            ### Step 2.1.1. 计算所有原子距该中心原子的距离
            # shape = (3,) -> (1, 3)
            tmp_center_coord = supercell_coords[tmp_center_idx].reshape(1, 3)
            # shape = (num_supercell, 3)
            tmp_relative_coords = supercell_coords - tmp_center_coord
            # shape = (num_supercell,)
            tmp_distances = np.linalg.norm(tmp_relative_coords, axis=1)
            
            ### Step 2.1.2. 将 `supercell 中所有原子的索引`按照距中心原子距离的远近排序（这个索引指的是在supercell中的索引）
            tmp_num_nbrs = np.count_nonzero(tmp_distances<=rcut)    # 该中心原子在截断半径内的近邻原子数 (包括自身)
            
            if tmp_num_nbrs > max_num_nbrs:
                max_num_nbrs = tmp_num_nbrs
            tmp_sorted_nbr_idxs = np.argsort(tmp_distances)[:tmp_num_nbrs]
            #print(tmp_sorted_nbr[1000:1050])
            
            ### Step 2.1.3.
            nbr_atomic_numbers[tmp_i, :tmp_num_nbrs] = supercell_atomic_numbers[tmp_sorted_nbr_idxs]
            nbr_distances[tmp_i, :tmp_num_nbrs] = tmp_distances[tmp_sorted_nbr_idxs]
            nbr_coords[tmp_i, :tmp_num_nbrs, :] = supercell_coords[tmp_sorted_nbr_idxs, :]
        
        ### Step 3. 
        nbr_atomic_numbers = nbr_atomic_numbers[:, :max_num_nbrs]
        nbr_distances = nbr_distances[:, :max_num_nbrs]
        nbr_coords = nbr_coords[:, :max_num_nbrs, :]
        
        return nbr_atomic_numbers, nbr_distances, nbr_coords