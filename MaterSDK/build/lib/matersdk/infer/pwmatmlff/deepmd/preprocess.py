import numpy as np
from typing import List

from ....io.publicLayer.structure import DStructure
from ....io.publicLayer.neigh import StructureNeighborsUtils
from ....io.publicLayer.neigh import StructureNeighborsDescriptor
from ....feature.deepmd.premise import DpFeaturePairPremiseDescriptor
from ....feature.deepmd.preprocess import TildeRNormalizer


class InferPreprocessor(object):
    def __init__(
                self,
                structure:DStructure,
                rcut:float,
                rcut_smooth:float,
                scaling_matrix:List[int],
                davg:np.ndarray,
                dstd:np.ndarray,
                center_atomic_numbers:List[int],
                nbr_atomic_numbers:List[int],
                max_num_nbrs:List[int],
                reformat_mark:bool=True,
                coords_are_cartesian:bool=True,
                ):
        '''
        Description
        -----------
            1. 为了适配 Siyu's code 做出inference，需要根据`max_num_nbrs`对以下数据进行zero-padding:
                1. ImageDR: 
                    - Name it as `rc`
                    - shape = (num_center, max_num_nbrs, 3)
                2. Ri:
                    - Name it as `tildeR`
                    - shape = (num_center, max_num_nbrs, 4)
                3. Ri_d:
                    - Name it as `tildeR_derivative`
                    - shape = (num_center, max_num_nbrs, 4, 3)
                4. list_neigh: 
                    - Name it as `list_neigh`
                    - shape = (num_center, max_num_nbrs)
                5. natoms_img:
                    - Name it as `natoms_img`
                    - shape = (1 + num_elements,)
        '''
        self.structure = structure
        self.center_atomic_numbers = center_atomic_numbers
        self.nbr_atomic_numbers = nbr_atomic_numbers
        self.max_num_nbrs = max_num_nbrs
        
        self.struct_nbr = StructureNeighborsDescriptor.create(
                'v1',
                structure=self.structure,
                rcut=rcut,
                scaling_matrix=scaling_matrix,
                reformat_mark=reformat_mark,
                coords_are_cartesian=coords_are_cartesian     
        )
        self.dp_pair_premise = DpFeaturePairPremiseDescriptor.create(
                'v1',
                structure_neighbors=self.struct_nbr
        )
        self.tildeR_normalizer = TildeRNormalizer(
                rcut=rcut,
                rcut_smooth=rcut_smooth,
                center_atomic_numbers=center_atomic_numbers,
                nbr_atomic_numbers=nbr_atomic_numbers,
                max_num_nbrs=max_num_nbrs,
                scaling_matrix=scaling_matrix,
                davgs=davg,
                dstds=dstd
        )
    
    
    def expand_rc(self):
        rcs_lst = []
        for tmp_center_an in self.center_atomic_numbers:
            tmp_center_rcs_lst = []
            for tmp_nbr_an_idx, tmp_nbr_an in enumerate(self.nbr_atomic_numbers):
                tmp_rc = self.dp_pair_premise.expand_rc(
                            center_atomic_number=tmp_center_an,
                            nbr_atomic_number=tmp_nbr_an,
                            max_num_nbrs=self.max_num_nbrs[tmp_nbr_an_idx]
                )
                tmp_center_rcs_lst.append(tmp_rc)
            tmp_center_rcs_array = np.concatenate(tmp_center_rcs_lst, axis=1)
            rcs_lst.append(tmp_center_rcs_array)
        rcs_array = np.concatenate(rcs_lst, axis=0)
        rc = np.expand_dims(rcs_array, axis=0)
        return rc
    
    
    def expand_tildeR(self):
        Ri, Ri_d = self.tildeR_normalizer.normalize(structure=self.structure)
        # shape = (1, num_centers, max_num_nbrs, 4)
        Ri = np.expand_dims(Ri, axis=0)
        # shape = (1, num_centers, max_num_nbrs, 4, 3)
        Ri_d = np.expand_dims(Ri_d, axis=0)
        return Ri, Ri_d
    
    
    def expand_list_neigh(self):
        list_neigh_ = StructureNeighborsUtils.get_nbrs_indices(
                struct_nbr=self.struct_nbr,
                center_atomic_numbers=self.center_atomic_numbers,
                nbr_atomic_numbers=self.nbr_atomic_numbers,
                max_num_nbrs=self.max_num_nbrs
        )
        list_neigh = np.expand_dims(list_neigh_, axis=0)
        
        return list_neigh
    
    
    def expand_natoms_img(self):
        natoms = self.structure.get_natoms()
        natoms_img = natoms_img = np.repeat(natoms[np.newaxis, :], 1, axis=0)
        return natoms_img