import numpy as np
from typing import List
from abc import ABC, abstractclassmethod

from ...io.publicLayer.neigh import StructureNeighborsV1
from .se_pair import (DpseTildeRPairDescriptor,
                    DpseTildeRPairV1)


class DpseTildeRBase(ABC):
    @abstractclassmethod
    def get_tildeR(self):
        pass
    

class DpseTildeRDescriptor(object):
    registry = {}
    
    @classmethod
    def register(cls, name:str):
        def wrapper(subclass:DpseTildeRBase):
            cls.registry[name] = subclass
        return wrapper
    
    @classmethod
    def create(cls, name:str, *args, **kwargs):
        subclass = cls.registry[name]
        if subclass is None:
            raise ValueError(f"No DpseTildeR registered with name {name}")
        return subclass(*args, **kwargs)



@DpseTildeRDescriptor.register("v1")
class DpseTildeRPairV1(DpseTildeRBase):
    '''
    Description
    -----------
        1. 
    '''
    def __init__(
                self,
                structure_neighbors:StructureNeighborsV1,
                center_atomic_numbers_lst:List[int],
                nbr_atomic_numbers_lst:List[int],
                sel:List[int],
                rcut:float,
                rcut_smooth:float):
        '''
        Note
        ----
            1. `nbr_atomic_numbers_lst` 与 `sel` 需要对应
        '''
        self.structure_neighbors = structure_neighbors
        self.center_atomic_numbers_lst = center_atomic_numbers_lst
        self.nbr_atomic_numbers_lst = nbr_atomic_numbers_lst
        self.sel = sel
        self.rcut = rcut
        self.rcut_smooth = rcut_smooth
    
    
    def get_tildeR(self):
        '''
        Description
        -----------
            1. 
        '''
        tilde_rs_lst = []
        for tmp_center_an in self.center_atomic_numbers_lst:
            for tmp_nbr_i, tmp_nbr_an in enumerate(self.nbr_atomic_numbers_lst):
                tmp_sel = self.sel[tmp_nbr_i]
                dpse_tildeR_pair = DpseTildeRPairDescriptor.create(
                                'v1',
                                self.structure_neighbors,
                                tmp_center_an,
                                tmp_nbr_an,
                                self.rcut,
                                self.rcut_smooth)
                tmp_tilde_r = dpse_tildeR_pair.dp_feature_pair_tildeR
                tmp_tilde_r_ext = np.zeros(
                    (tmp_tilde_r.shape[0], tmp_sel, tmp_tilde_r.shape[2])
                )
                tmp_tilde_r_ext[:, :tmp_tilde_r.shape[1], :] = tmp_tilde_r
                '''
                (8, 20, 4)
                (8, 15, 4)
                (4, 20, 4)
                (4, 15, 4)
                '''
                #print(tmp_center_an, tmp_nbr_an, tmp_tilde_r_ext.shape)
                #print(tmp_tilde_r_ext)
                tilde_rs_lst.append(tmp_tilde_r_ext.flatten())
            
        tilde_r_tot = np.concatenate(tilde_rs_lst)
        return tilde_r_tot