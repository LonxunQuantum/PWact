import numpy as np 
from typing import List, Dict

from ....io.publicLayer.structure import DStructure
from ....io.publicLayer.neigh import StructureNeighborsDescriptor
from ....feature.deepmd.se_pair import DpseTildeRPairDescriptor
from ....feature.deepmd.preprocess import TildeRPairNormalizer
from ....infer.pwmatmlff.deepmd.extractor import FFExtractor


class FFInfer(object):
    def __init__(
                self,
                hdf5_path:str,
                rcut:float,
                rcut_smooth:float,
                max_num_nbrs_dict: Dict[int, int],
                davgs_dict: Dict[int, np.ndarray],
                dstds_dict: Dict[int, np.ndarray]):
        '''
        Parameters
        ----------
            1. hdf5_path: str
                - The path of hdf5 file.
            2. rcut: float
                - 
            3. rcut_smooth: float
                - 
            4. max_num_nbrs_dict: Dict[int, int]
                - key: 近邻原子的原子序数
                - value: key原子的最大近邻数目
                - e.g. {3: 100, 14: 80}
            5. davgs_dict: Dict[int, np.ndarray]
                - int: 中心元素的原子序数
                - value.shape = (num_types, 4)
                    - `num_types`: 元素的种类
                    - `davg.shape = (4,)`
            6. dstds_dict: Dict[int, np.ndarray]
                - int: 中心元素的原子序数
                - value.shape = (num_types, 4)
                    - `num_types`: 元素的种类
                    - `davg.shape = (4,)`
        '''
        self.model_params_dict = self._load_model_params_dict(hdf5_path=hdf5_path)

        self.rcut = rcut
        self.rcut_smooth = rcut_smooth
        self.max_num_nbrs_dict = max_num_nbrs_dict
        self.davgs_dict = davgs_dict
        self.dstds_dict = dstds_dict
        #self.normalizer = TildeRPairNormalizer(davg=davg, dstd=dstd)
    
    
    def _load_model_params_dict(self, hdf5_path:str):
        '''
        Description
        -----------
            1. load hdf5 file to get params of Neural network of deepmd
        
        Parameters
        ----------
            1. hdf5_dict: Dict[str, np.ndarray]
                - 神经网络的参数
        '''
        hdf5_dict:Dict[str, np.ndarray] = FFExtractor.get_hdf5_dict(hdf5_path=hdf5_path)
        return hdf5_dict
    
    
    def calc_tildeR(
                self,
                structure:DStructure,
                scaling_matrix:List[int]
                ):
        '''
        Description
        -----------
            1. 根据结构文件提取其 Deepmd 的feature
            
        Parameters
        ----------
            1. structure: DStructure
            2. scaling_matrix: List[int]
                - e.g. [3, 3, 3]
        
        Return
        ------
            1. tildeR_dict: Dict[str, np.ndarray]:
                - e.g. {
                        "3_3": np.ndarray, 
                        "3_14": np.ndarray, 
                        ...}
        '''
        tildeR_dict:Dict[str, np.ndarray] = {}
        
        ### Step 1. 得到 DStructure 的信息
        ### Step 1.1. atomic_numbers_lst
        atomic_numbers_lst = self._get_atomic_numbers(structure=structure)
        
        ### Step 2. 计算 tildeR in DeepPot-SE
        struct_nbr = StructureNeighborsDescriptor.create(
                "v1",
                structure=structure,
                rcut=self.rcut,
                scaling_matrix=scaling_matrix,
                reformat_mark=True,
                coords_are_cartesian=True)
        for tmp_center_an in atomic_numbers_lst:
            for tmp_nbr_an in atomic_numbers_lst:
                tmp_dpse_tildeR_pair = DpseTildeRPairDescriptor.create(
                        'v1',
                        structure_neighbors=struct_nbr,
                        center_atomic_number=tmp_center_an,
                        nbr_atomic_number=tmp_nbr_an,
                        rcut=self.rcut,
                        rcut_smooth=self.rcut_smooth)
                tildeR_dict.update(
                    {
                    "{0}_{1}".format(tmp_center_an, tmp_nbr_an): \
                        tmp_dpse_tildeR_pair.get_tildeR(max_num_nbrs=self.max_num_nbrs_dict[tmp_nbr_an])
                    }
                )
                
        ### Step 3. Normalize
        
        
        return tildeR_dict
        
    
    def _get_atomic_numbers(self, structure:DStructure):
        '''
        Description
        -----------
            1. 得到 `DStructure` 的原子序数 (不重复)
        '''
        atomic_numbers_lst = [tmp_specie.Z for tmp_specie in structure.species]
        atomic_numbers_set = set(atomic_numbers_lst)
        return list(atomic_numbers_set)
    
    
    def _normalize(self):
        '''
        Description
        -----------
            1. 

        Parameters
        ----------
            1. davg: 
            2. dstd: 
        '''
        pass