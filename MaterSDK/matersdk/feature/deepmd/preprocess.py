import numpy as np
import h5py
from typing import Union, List, Dict

from ...data.deepmd.data_system import DpLabeledSystem
from ...io.publicLayer.structure import DStructure
from ...io.publicLayer.neigh import StructureNeighborsDescriptor
from ...feature.deepmd.se_pair import DpseTildeRPairDescriptor


class TildeRNormalizer(object):
    '''
    Description
    -----------
        1. 计算 $\tilde{R}$ + normalize 一条龙
        2. 可以设置中心原子为不同的元素
            - e.g. 计算下列 pair 的 `davg`, `dstd`
                1) Li-Li/Si 的 $\tilde{R}$
                2) Si-Li/Si 的$\tilde{R}$
        3. 在 `TildeRNormalizer` 内部会调用：
            1. `TildeRPairNormalizer`
                - 
            2. `NormalizerPremise`
                - 
    '''
    def __init__(
            self,
            rcut:float,
            rcut_smooth:float,
            center_atomic_numbers:Union[List[int], np.ndarray],
            nbr_atomic_numbers:Union[List[int], np.ndarray],
            max_num_nbrs:List[int],
            scaling_matrix:List[int],
            davgs:np.ndarray,
            dstds:np.ndarray
            ):
        '''
        Parameters
        ----------
            1. rcut: float
                - DeepPot-SE 的截断半径
            2. rcut_smooth: float
                - DeepPot-SE 的 smooth cutoff
            3. center_atomic_numbers: List[int]
                - 中心原子的原子序数，按照从小到大排序
            4. nbr_atomic_numbers: List[int]
                - 近邻原子的原子序数，按照从小到大排序
            5. max_num_nbrs: List[int]
                - 中心原子近邻的nbr元素最多的原子数目
                - Note: 与 nbr_atomic_numbers 一一对应
            6. scaling_matrix: List[int]
                - 
            7. davgs: Union[np.ndarray, bool]
                - shape = (num_types, 4)
            8. dstds: Union[np.ndarray, bool]
                - shape = (num_types, 4)
            
        Note
        ----    
            1. 有两种方法初始化 `TildeRNormalizer`
                1) dp_labeled_system + structure_indices
                2) davgs + dstds
        '''
        self.rcut = rcut
        self.rcut_smooth = rcut_smooth
        self.center_atomic_numbers = center_atomic_numbers
        self.nbr_atomic_numbers = nbr_atomic_numbers
        self.max_num_nbrs = max_num_nbrs
        self.scaling_matrix = scaling_matrix
        
        ### Step 1. Get the `davgs` and `dstds`
        self.davgs = davgs
        self.dstds = dstds
        
        assert (davgs.shape[0] == len(self.center_atomic_numbers))
        assert (davgs.shape[1] == 4)
        assert (dstds.shape[0] == len(self.center_atomic_numbers))
        assert (dstds.shape[1] == 4)
    
    
    def __str__(self):
        return self.__repr__()
    
    
    def __repr__(self):
        print("{0:*^80s}".format(" TildeRNormalizer Summary "))
        
        print("\t * {0:26s}: {1:14f}".format("rcut", self.rcut))
        print("\t * {0:26s}: {1:14f}".format("rcut_smooth", self.rcut_smooth))
        print("\t * {0:26s}:\t".format("center_atomic_numbers:"), self.center_atomic_numbers)
        print("\t * {0:26s}:\t".format("nbr_atomic_numbers:"), self.nbr_atomic_numbers)
        print("\t * {0:26s}:\t".format("max_num_nbrs"), self.max_num_nbrs)
        print("\t * {0:26s}:\t".format("scaling_matrix"), self.scaling_matrix)
        print("\t * {0:26s}:\t".format("davgs"))
        print(self.davgs)
        print("\t * {0:26s}:\t".format("dstds"))
        print(self.dstds)        
        
        print("{0:*^80s}".format("**"))
        return ""
    
    
    @staticmethod
    def calc_stats(
                dp_labeled_system:DpLabeledSystem,
                structure_indices:List[int],
                rcut:float,
                rcut_smooth:float,
                center_atomic_numbers:List[int],
                nbr_atomic_numbers:List[int],
                max_num_nbrs:List[int],
                scaling_matrix:List[int],
                ):
        '''
        Description
        -----------
            1. 如果初始化的时候使用 `dp_labeled_system` 和 `structure_indices`，则会调用这个函数
        
        Parameters
        ----------
            1. dp_labeled_system: DpLabeledSystem
                - 
            2. structure_indices: List[int]
                - 
        
        Return
        ------
            1. davgs: np.ndarray
                - .shape = (num_types, 4)
            2. dstds: np.ndarray
                - .shape = (num_types, 4)
        '''
        ### Step 1. 
        #       e.g. {
        #            3: (48, 1800, 4)
        #           14: (24, 1800, 4)
        #}
        davgs_lst = []
        dstds_lst = []
        for tmp_center_an in center_atomic_numbers:
            ### Step 1.1. Calcuate `tildeRs_array`
            # e.g.
            #   Li .shape = (48, 1800, 4)
            #   Si .shape = (24, 1800, 4)
            tmp_tildeRs_array = NormalizerPremise.concat_tildeRs4calc_stats(
                    dp_labeled_system=dp_labeled_system,
                    structure_indices=structure_indices,
                    rcut=rcut,
                    rcut_smooth=rcut_smooth,
                    center_atomic_number=tmp_center_an,
                    nbr_atomic_numbers=nbr_atomic_numbers,
                    max_num_nbrs=max_num_nbrs,
                    scaling_matrix=scaling_matrix
            )
            
            ### Step 1.2. Calculate `davg`, `dstd`
            tmp_normalizer = TildeRPairNormalizer(tildeRs_array=tmp_tildeRs_array)
            # tmp_normalizer.davg.shape = (1, 4)
            # tmp_normalizer.dstd.shape = (1, 4)
            davgs_lst.append(tmp_normalizer.davg)
            dstds_lst.append(tmp_normalizer.dstd)
        
        ### Step 2.
        # shape = (num_types, 4)
        davgs = np.concatenate(davgs_lst, axis=0)
        # shape = (num_types, 4)
        dstds = np.concatenate(dstds_lst, axis=0)
        
        return davgs, dstds
    

    def to(self, hdf5_file_path:str):
        h5_file = h5py.File(hdf5_file_path, 'w')
        
        h5_file.create_dataset("rcut", data=self.rcut)
        h5_file.create_dataset("rcut_smooth", data=self.rcut_smooth)
        h5_file.create_dataset("center_atomic_numbers", data=self.center_atomic_numbers)
        h5_file.create_dataset("nbr_atomic_numbers", data=self.nbr_atomic_numbers)
        h5_file.create_dataset("max_num_nbrs", data=self.max_num_nbrs)
        h5_file.create_dataset("scaling_matrix", data=np.array(self.scaling_matrix))
        h5_file.create_dataset("davgs", data=self.davgs)
        h5_file.create_dataset("dstds", data=self.dstds)
        
        h5_file.close()
        
        
    @classmethod
    def from_file(cls, hdf5_file_path:str):
        ### Step 1. Extract information from hdf5 file
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        rcut = hdf5_file["rcut"][()]
        rcut_smooth = hdf5_file["rcut_smooth"][()]
        center_atomic_numbers = hdf5_file["center_atomic_numbers"][()]
        nbr_atomic_numbers = hdf5_file["nbr_atomic_numbers"][()]
        max_num_nbrs = hdf5_file["max_num_nbrs"][()]
        scaling_matrix = hdf5_file["scaling_matrix"][()]
        davgs = hdf5_file["davgs"][()]
        dstds = hdf5_file["dstds"][()]
        hdf5_file.close()
        
        ### Step 2. Initialize the `TildeRNormailzer`
        tilde_r_normalizer = cls(
                        rcut=rcut,
                        rcut_smooth=rcut_smooth,
                        center_atomic_numbers=center_atomic_numbers,
                        nbr_atomic_numbers=nbr_atomic_numbers,
                        max_num_nbrs=max_num_nbrs,
                        scaling_matrix=scaling_matrix,
                        davgs=davgs,
                        dstds=dstds
        )
        
        return tilde_r_normalizer
    
    
    @classmethod
    def from_dp_labeled_system(
                    cls,
                    dp_labeled_system:DpLabeledSystem,
                    structure_indices:List[int],
                    rcut:float, 
                    rcut_smooth:float,
                    center_atomic_numbers:List[int],
                    nbr_atomic_numbers:List[int],
                    max_num_nbrs:List[int],
                    scaling_matrix:List[int]):
        '''
        Descroption
        -----------
            1. 由一个 DpLabeledSystem 初始化 TildeRNormalizer
        
        Parameters
        ----------
            1. 
        '''
        davgs, dstds = TildeRNormalizer.calc_stats(
                    dp_labeled_system=dp_labeled_system,
                    structure_indices=structure_indices,
                    rcut=rcut,
                    rcut_smooth=rcut_smooth,
                    center_atomic_numbers=center_atomic_numbers,
                    nbr_atomic_numbers=nbr_atomic_numbers,
                    max_num_nbrs=max_num_nbrs,
                    scaling_matrix=scaling_matrix
        )
        tilde_r_normalizer = cls(
                        rcut=rcut,
                        rcut_smooth=rcut_smooth,
                        center_atomic_numbers=center_atomic_numbers,
                        nbr_atomic_numbers=nbr_atomic_numbers,
                        max_num_nbrs=max_num_nbrs,
                        scaling_matrix=scaling_matrix,
                        davgs=davgs,
                        dstds=dstds)

        return tilde_r_normalizer
    
    
    
    def _normalize(self, structure:DStructure):
        '''
        Description
        -----------
            1.
        
        Parameters
        ----------
            1. structure: DStructure
                - 
        
        Return
        ------
            1. tildeR_dict: Dict[str, np.ndarray]
                - e.g. {
                        "3_3": np.ndarray,  # shape = (num_centers, max_num_nbrs, 4)
                        "3_14": np.ndarray, # shape = (num_centers, max_num_nbrs, 4)
                        "14_3": np.ndarray, # shape = (num_centers, max_num_nbrs, 4)
                        "14_14": np.ndarray # shape = (num_centers, max_num_nbrs, 4)
                    }
                - concat `tildeR_dict["3_3"]` and `tildeR_dict["3_14"]` -> Li's Environment matrix 
            
            2. tildeR_derivative_dict: Dict[str, np.ndarray]
                - e.g. {
                        "3_3": np.ndarray,  # shape = (num_centers, max_num_nbrs, 4, 3)
                        "3_14": np.ndarray, # shape = (num_centers, max_num_nbrs, 4, 3)
                        "14_3": np.ndarray, # shape = (num_centers, max_num_nbrs, 4, 3)
                        "14_14": np.ndarray # shape = (num_centers, max_num_nbrs, 4, 3)
                    }
        '''
        tildeR_dict:Dict[str, np.ndarray] = {}
        tildeR_derivative_dict:Dict[str, np.ndarray] = {}
        ### Step 1. 计算新结构的 `StructureNeighbors`
        struct_neigh = StructureNeighborsDescriptor.create(
                    'v1',
                    structure=structure,
                    rcut=self.rcut,
                    scaling_matrix=self.scaling_matrix,
                    reformat_mark=True,
                    coords_are_cartesian=True)
        
        ### Step 2. 计算 Normalized 的 $\tilde{R}$
        for tmp_idx_center_an, tmp_center_an in enumerate(self.center_atomic_numbers):
            for tmp_idx_nbr_an, tmp_nbr_an in enumerate(self.nbr_atomic_numbers):
                ### Step 2.1. 计算 Environment Matrix -- $\tilde{R}$ 
                dpse_tildeR_pair = DpseTildeRPairDescriptor.create(
                            'v1',
                            structure_neighbors=struct_neigh,
                            center_atomic_number=tmp_center_an,
                            nbr_atomic_number=tmp_nbr_an,
                            rcut=self.rcut,
                            rcut_smooth=self.rcut_smooth
                )
                ### 计算 Environment Matrix
                tmp_tilde_R = dpse_tildeR_pair.get_tildeR(
                                max_num_nbrs=self.max_num_nbrs[tmp_idx_nbr_an])
                ### 计算 derivative of Environment Matrix with respect to x, y, z
                tmp_tilde_R_derivative = dpse_tildeR_pair.calc_derivative(
                                max_num_nbrs=self.max_num_nbrs[tmp_idx_nbr_an])
                
                ### Step 2.2. Normalize Environment Matrix -- $\tilde{R}$
                tmp_davg = self.davgs[tmp_idx_center_an]
                tmp_dstd = self.dstds[tmp_idx_center_an]
                tmp_tilde_r_pair_normalizer = TildeRPairNormalizer(
                            davg=tmp_davg,
                            dstd=tmp_dstd
                )
                tmp_normalized_tildeR = tmp_tilde_r_pair_normalizer.normalize(tildeRs_array=tmp_tilde_R)
                tildeR_dict.update({"{0}_{1}".format(tmp_center_an, tmp_nbr_an): tmp_normalized_tildeR})
                
                ### Step 2.3. Scaling the derivative of Environment Matrix (Rij) with respect to x, y, z
                tmp_scaled_tildeR_derivative = tmp_tilde_r_pair_normalizer.normalize_derivative(
                                                    tildeR_derivatives_array=tmp_tilde_R_derivative)
                tildeR_derivative_dict.update({"{0}_{1}".format(tmp_center_an, tmp_nbr_an): tmp_scaled_tildeR_derivative})
                
        return tildeR_dict, tildeR_derivative_dict
        
        
    def normalize(self, structure):
        '''
        Description
        -----------
            1. self._normalize() 返回：
                1. tildeRs_dict: 
                    1) (48, 100, 4)
                    2) (48, 80, 4)
                    3) (24, 100, 4)
                    4) (24, 80, 4)
                2. tildeR_derivatives_dict: 
                    1) (48, 100, 4, 3)
                    2) (48, 80, 4, 3)
                    3) (24, 100, 4, 3)
                    4) (24, 80, 4, 3)
            2. 合并为： 
                1. tildeR
                    1) (72, 180, 4)
                2. tildeR_derivative:
                    1) (72, 180, 4, 3)
        '''
        tildeRs_dict, tildeR_derivatives_dict = self._normalize(structure=structure) 

        tildeRs_lst = []
        tildeR_derivatives_lst = []
        for tmp_center_an in self.center_atomic_numbers:
            tmp_center_tildeRs_lst = []
            tmp_center_tildeR_derivatives_lst = []
            for tmp_nbr_an in self.center_atomic_numbers:
                tmp_key = "{0}_{1}".format(tmp_center_an, tmp_nbr_an)
                tmp_center_tildeRs_lst.append(tildeRs_dict[tmp_key])
                tmp_center_tildeR_derivatives_lst.append(tildeR_derivatives_dict[tmp_key])
            
            tmp_center_tildeRs_array = np.concatenate(tmp_center_tildeRs_lst, axis=1)
            tildeRs_lst.append(tmp_center_tildeRs_array)
            tmp_center_tildeR_derivatives_array = np.concatenate(tmp_center_tildeR_derivatives_lst, axis=1)
            tildeR_derivatives_lst.append(tmp_center_tildeR_derivatives_array)
        
        tildeR = np.concatenate(tildeRs_lst, axis=0)
        tildeR_derivative = np.concatenate(tildeR_derivatives_lst, axis=0)
        
        return tildeR, tildeR_derivative
        
                


class TildeRPairNormalizer(object):
    '''
    Description
    -----------
        1. 适用于单个结构、单个中心原子！
        2. 中心原子确定
            - e.g. 计算下列 pair 的 `davg`, `dstd`
                1) Li-Li/Si 的 $\tilde{R}$
                2) Si-Li/Si 的 $\tilde{R}$
                
    '''
    def __init__(self,
                tildeRs_array:Union[np.ndarray, bool]=False,
                davg:Union[np.ndarray, bool]=False,
                dstd:Union[np.ndarray, bool]=False):
        '''
        Description
        -----------
            1. Calculate the `davg` and `dstd` of Environment matrix ($\widetilde{R}^i = (s, sx/r, sy/r, sz/r)$)
            2. Normalize the Environment matrix ($\widetilde{R}^i = (s, sx/r, sy/r, sz/r)$)

        Parameters
        ----------
            1. tildeRs_array: 
                - `tildeRs_array.shape = (48, 80, 4) concat (48, 100, 4) = (48, 180, 4)`
                    ```
                    struct_nbr = StructureNeighborsDescriptor.create(
                                    'v1',
                                    structure=structure,
                                    rcut=rcut,
                                    scaling_matrix=scaling_matrix,
                                    reformat_mark=reformat_mark,
                                    coords_are_cartesian=coords_are_cartesian)
                    dpse_tildeR_pair_Li = DpseTildeRPairDescriptor.create(
                                    'v1',
                                    structure_neighbors=struct_nbr,
                                    center_atomic_number=center_atomic_number,
                                    nbr_atomic_number=3,
                                    rcut=rcut,
                                    rcut_smooth=rcut_smooth)
                    #print(dpse_tildeR_pair.dp_feature_pair_tildeR)
                    tildeRs_array_Li = dpse_tildeR_pair_Li.get_tildeR(max_num_nbrs=100)
                    
                    dpse_tildeR_pair_Si = DpseTildeRPairDescriptor.create(
                                    'v1',
                                    structure_neighbors=struct_nbr,
                                    center_atomic_number=center_atomic_number,
                                    nbr_atomic_number=nbr_atomic_number,
                                    rcut=rcut,
                                    rcut_smooth=rcut_smooth)
                    #print(dpse_tildeR_pair.dp_feature_pair_tildeR)
                    tildeRs_array_Si = dpse_tildeR_pair_Si.get_tildeR(max_num_nbrs=80)
                    # (48, 100, 4) + (48, 80, 4) = (48, 180, 4)
                    tildeRs_array = np.concatenate([tildeRs_array_Li, tildeRs_array_Si], axis=1)
                    ```
            2. davg: np.ndarray 
                - `davg.shape = (1, 4)`
            3. dstd: np.ndarray
                - `dstd.shape = (1, 4)`
        '''
        # shape = (1, 4)
        if (davg is not False) and (dstd is not False):
            self.davg = davg
            self.dstd = dstd
        elif (tildeRs_array is not False):
            # shape: (num_frames, num_centers, max_num_nbrs, 4)     e.g. (48, 26, 4)
            #   ->
            # shape: (num_frames * num_centers * max_num_nbrs, 4)   e.g. (1248, 4)
            self.davg, self.dstd = self.calc_stats( tildeRs_array.reshape(-1, 4) )
        else:
            raise ValueError("You should check davg and dstd!")
    
    
    def calc_stats(self, tildeRs_array:np.ndarray):
        '''
        Description
        -----------
            1. 计算 DeepPot-SE 中 TildeR 的平均值(`avg`)和方差(`std`)
            2. 如果初始化的时候使用 `dp_labeled_system` 和 `structure_indices`，则会调用这个函数
        
        Parameters
        ----------
            1. tildeRs_array: np.ndarray
                - .shape = (num_frames * num_centers * max_num_nbrs, 4)
        '''
        ### Step 1. 分别获取径向信息(`info_radius`)和角度信息(`info_angles`)
        # shape: (num_frames * num_centers * max_num_nbrs, 1)
        info_radius = tildeRs_array[:, 0].reshape(-1, 1)
        # shape: (num_frames * num_centers * max_num_nbrs, 3)
        info_angles = tildeRs_array[:, 1:].reshape(-1, 3)
        
        ### Step 2. 分别获取径向信息(`info_radius`)和角度信息(`info_angles`)的一些量:
        #     1) sum: 求和
        #     2) sum^2: 平方和 -- 先平方后求和
        #     3) total_num_pairs: num_centers * max_num_nbrs
        ### Step 2.1. sum: 求和
        sum_info_radius = np.sum(info_radius)
        sum_info_angles = np.sum(info_angles) / 3.0
        
        ### Step 2.2. sum^2: 平方和 -- 先平方后求和
        sum2_info_radius = np.sum(
                    np.multiply(info_radius, info_radius)
        )
        sum2_info_angles = np.sum(
                    np.multiply(info_angles, info_angles)
        ) / 3.0
        
        ### Step 2.3. total_num_pairs.shape: num_centers * max_num_nbrs
        total_num_pairs = info_radius.shape[0] # Error: np.count_nonzero(info_radius.flatten() != 0.)

        
        ### Step 3. 计算平均值 -- davg_unit
        davg_unit = [sum_info_radius / (total_num_pairs + 1e-15), 0, 0, 0]
        # shape = (1, 4)
        davg_unit = np.array(davg_unit).reshape(-1, 4)
        
        
        ### Step 4. 计算方差 -- dstd_unit
        dstd_unit = [
            self._calc_std(sum2_value=sum2_info_radius, sum_value=sum_info_radius, N=total_num_pairs),
            self._calc_std(sum2_value=sum2_info_angles, sum_value=sum_info_angles, N=total_num_pairs),
            self._calc_std(sum2_value=sum2_info_angles, sum_value=sum_info_angles, N=total_num_pairs),
            self._calc_std(sum2_value=sum2_info_angles, sum_value=sum_info_angles, N=total_num_pairs)
        ]
        # shape = (1, 4)
        dstd_unit = np.array(dstd_unit).reshape(-1, 4)
        
        return davg_unit, dstd_unit
        
    
    def _calc_std(self, sum2_value:float, sum_value:float, N:int):
        '''
        Description
        -----------
            1. 计算标准差
        
        Parameters
        ----------
            1. sum2_value: float
                - sum2_value = \sum_i^N{x_i^2}，先平方后求和
            2. sum_value: float
                - sum_value  = \sum_i^N{x_i}
        '''
        if (N == 0):
            return 1e-2
        std = np.sqrt(
                sum2_value / N - np.multiply(sum_value/N, sum_value/N)
        )
        if np.abs(std) < 1e-2:
            std = 1e-2
        return std
        
    
    def normalize(self, tildeRs_array:np.ndarray):
        '''
        Description
        -----------
            1. 将 tildeRs_array(多个结构的DeepPot-SE特征) 或 tildeR_array(单个结构的DeepPot-SE特征) 归一化
        
        Parameters
        ----------
            1. tildeRs_array: np.ndarray
                - shape = 
                    1. (num_frames, num_centers, max_num_nbrs, 4)
                    2. (num_centers, max_num_nbrs, 4)
                
        
        Note
        ----    
            1. You can input environment matrix for `single frame` or `many frames`
                - single frame: .shape = (num_centers, max_num_nbrs, 4) 
                - many frames : .shape = (num_frames, num_centers, max_num_nbrs, 4)
        '''
        if (len(tildeRs_array.shape) == 3):
            # single frame: tildeR.shape = (num_centers, max_num_nbrs, 4)
            # (1, 4) -> (1, 1, 4)
            davg = self.davg.reshape(1, 1, 4)
            dstd = self.dstd.reshape(1, 1, 4)
        elif (len(tildeRs_array.shape) == 4):
            # many frames: tildeR.shape = (num_frames, num_centers, max_num_nbrs, 4)
            # (1, 4) -> (1, 1, 1, 4)
            davg = self.davg.reshape(1, 1, 1, 4)
            dstd = self.dstd.reshape(1, 1, 1, 4)
        
        ### Normalize the environment matrix (Rij)
        result = (tildeRs_array - davg) / dstd
        
        return result
    
    
    def normalize_derivative(self, tildeR_derivatives_array:np.ndarray):
        '''
        Description
        -----------
            1. 将 tildeR_derivatives_array (单个结构DeepPot-SE的导数) 或 tildeR_detivatives_array (多个结构DeepPot-SE的导数) 进行归一化
            
        Parameters
        ----------
            1. tildeR_derivatives_array: np.ndarray
                - shape = 
                    1. Single frame -- (num_centers, max_num_nbrs, 4, 3)
                    2. Many frames  -- (num_frames, num_centers, max_num_nbrs, 4, 3)
        '''
        if (len(tildeR_derivatives_array.shape) == 4):
            # single frame, tildeR_derivatives_array.shape = (num_centers, max_num_nbrs, 4, 3)
            # (1, 4) -> (1, 1, 4, 1)
            dstd = self.dstd.reshape(1, 1, 4, 1)
        elif (len(tildeR_derivatives_array.shape) == 5):
            # many frames, tildeR_derivatives_array.shape = (num_frames, num_centers, max_num_nbrs, 4, 3)
            dstd = self.dstd.reshape(1, 1, 1, 4, 1)
        
        ### Normalize the derivative of environment matrix (Rij) with respect to x, y, z
        result = tildeR_derivatives_array / dstd
    
        return result




class NormalizerPremise(object):
    '''
    Static Member Function:
        1. concat_tildeRs4calc_stats:
            - 计算 Environment matrix (Rij) 的 statistic data (avg, std) 时使用
            - return shape = (48, 180 * 10, 4)
        2. concat_tildeRs4nromalize:
            - 归一化新结构的 Environment matrix (Rij) 时使用
            - return shape = (10, 48, 180, 4)
    '''
    @staticmethod
    def concat_tildeRs4calc_stats(
                dp_labeled_system:DpLabeledSystem,
                structure_indices:List[int],
                rcut:float, 
                rcut_smooth:float,
                center_atomic_number:int,
                nbr_atomic_numbers:List[int],
                max_num_nbrs:List[int],
                scaling_matrix:List[int]):
        '''
        Description
        -----------
            1. 中心原子的元素种类是确定的！！！
                - e.g. 计算多个结构的 Li-Li, Li-Si 的 $\tilde{R}$ 并合并
                    - Li-Li: (48, 100, 4) -- (num_centers, max_num_nbrs, 4)
                    - Li-Si: (48, 80, 4)  -- (num_centers, max_num_nbrs, 4)
                    - 取10个结构并计算`avg`和`std`。
                    - (48, 100, 4) + (48, 80, 4) -> (48, 180, 4) -- Environment Matrix
                    - Terminal Result = (48, 180 * num_frames, 4)
        
        Parameters
        ----------
            1. dp_labeled_system: DpLabeledSystem
                - 
            2. structure_indices: List[int]
                - 选取哪些结构，计算 statistic data -- avg, std
            3. rcut: float
                - rcutoff in DeepPot-SE
            4. rcut_smooth: float
                - rcutoff smooth in DeepPot-SE
            5. center_atomic_number: int
                - 中心原子的元素种类
            6. nbr_atomic_numbers: List[int]
                - 近邻原子的元素种类，所有近邻元素种类
            7. max_num_nbrs: List[int],
                - 最大近邻原子数，需要与 `nbr_atomic_numbers` 对应
            8. scaling_matrix: List[int]
                - 扩包倍数
            
        Return 
        ------ 
            1. tildeRs_array: np.ndarray
                - shape = (num_centers, num_frames * max_num_nbrs, 4)
            2. tildeR_derivatives_array: np.ndarray
                - shape = (num_centers, num_frames * max_num_nbrs, 4, 3)
        '''
        ### Step 1. 得到 DStructure 的列表
        structures_lst = [
            dp_labeled_system.structures_lst[tmp_idx] for tmp_idx in structure_indices
        ]
        
        ### Step 2. 合并获取 `tildeRs_array`, `tildeR_derivatives_array`
        # shape = (num_centers, num*frames * max_num_nbrs, 4)
        all_structures_tildeRs_lst = []    # 所有结构的 $\tilde{R}$
        for tmp_structure in structures_lst:
            tmp_struct_nbr = StructureNeighborsDescriptor.create(
                            'v1',
                            structure=tmp_structure,
                            rcut=rcut,
                            scaling_matrix=scaling_matrix,
                            reformat_mark=True,
                            coords_are_cartesian=True)
            
            tmp_all_nbrs_tildeRs_lst = []    # 同一结构，同一中心原子，不同近邻原子
            for tmp_idx_nbr_an, tmp_nbr_an in enumerate(nbr_atomic_numbers):
                # e.g. Li-Li : (48, 100, 4) -- (num_centers, max_num_nbrs ,4)
                # e.g. Li-Si : (48, 80,  4) -- (num_centers, max_num_nbrs, 4)
                tmp_nbr_tildeR = DpseTildeRPairDescriptor.create(
                            'v1',
                            structure_neighbors=tmp_struct_nbr,
                            center_atomic_number=center_atomic_number,
                            nbr_atomic_number=tmp_nbr_an,
                            rcut=rcut,
                            rcut_smooth=rcut_smooth).get_tildeR(
                                    max_num_nbrs=max_num_nbrs[tmp_idx_nbr_an])
                tmp_all_nbrs_tildeRs_lst.append(tmp_nbr_tildeR)
            
            # tmp_tildeR: Li-Li&Si
            # shape = (48, 180, 4)
            tmp_structure_tildeR = np.concatenate(tmp_all_nbrs_tildeRs_lst, axis=1)
            all_structures_tildeRs_lst.append(tmp_structure_tildeR)
            
        # tildeR_tot: all structures for Li-Li/Si
        # shape = (48, 180 * 10, 4)
        tildeRs_array = np.concatenate(all_structures_tildeRs_lst, axis=1)
        
        return tildeRs_array