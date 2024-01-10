from abc import ABC, abstractmethod
from typing import List
import copy
import numpy as np

from .acextractor import ACstrExtractor
from .lineLocator import LineLocator


class MVExtractor(object):
    def __init__(self, movement_path:str):
        self.mv_path = movement_path
        
        self.chunksizes:List[int] = self.get_chunksizes()   # set `self.num_frames`
        self.chunkslices:List[int] = self.get_chunkslices()
        self.virial_mark, self.eatoms_mark, self.magmoms_mark = \
                            self._find_extra_properties()            
    
    
    def get_num_frames(self):
        return self.num_frames
    
    
    def get_chunksizes(self):
        '''
        Description
        -----------
            1. 由于MOVEMENT 文件太大，因此将 MOVEMENT 的每一帧 (frame) 对应的内容，定义为一个chunk
            文件处理时 `一个chunk一个chunk处理`
            
        Return
        ------
            1. chunk_size: int 
                - 每一帧的行数
                - e.g. [225, 382, 382, 382, 382, 382, ...]
        
        Note
        ----
            1. chunksize 包括 `-------` 这一行
        '''
        chunksizes_lst:List[int] = []
        content = "--------------------------------------"
        aim_row_idxs = LineLocator.locate_all_lines(
                        file_path=self.mv_path, 
                        content=content)

        self.num_frames = len(aim_row_idxs)
        
        chunksizes_lst.append(aim_row_idxs[0])
        for idx in range(1, len(aim_row_idxs)):
            chunksizes_lst.append(aim_row_idxs[idx] - aim_row_idxs[idx-1])
        return chunksizes_lst
    
    
    def get_chunkslices(self):
        '''
        Return
        ------
            1. chunkslices: np.ndarrary
                - e.g. [     0    225    607    989   1371, ... ]
        '''
        chunksizes = copy.deepcopy(self.chunksizes)
        chunksizes.insert(0, 0)
        chunksizes = np.cumsum(chunksizes)
        
        return chunksizes
    
    
    def get_frame_str(self, fidx:int):
        frame_str:str = ""
        with open(self.mv_path, "r") as mv:
            for idx_line, line in enumerate(mv):
                if idx_line in range(self.chunkslices[fidx], self.chunkslices[fidx+1]):
                    frame_str += line
                elif idx_line >= self.chunkslices[fidx+1]:
                    break
        return frame_str
        
    
    def get_frame_info(self, fidx:int):
        infos = []
        frame_str = self.get_frame_str(fidx=fidx)
        ace_str_extractor = ACstrExtractor(atom_config_str=frame_str)
        
        box = ace_str_extractor.get_basis_vectors()
        types = ace_str_extractor.get_types()
        coords = ace_str_extractor.get_coords()
        etot = ace_str_extractor.get_etot()
        fatoms = ace_str_extractor.get_fatoms()
        
        infos.append( box )
        infos.append( types )
        infos.append( coords )
        infos.append( etot )
        infos.append( fatoms )
        
        if self.virial_mark:
            virial = ace_str_extractor.get_virial()
            infos.append( virial )
        if self.magmoms_mark:
            magmoms = ace_str_extractor.get_magmoms()
            infos.append( magmoms )
        if self.eatoms_mark:
            eatoms = ace_str_extractor.get_eatoms()
            infos.append( eatoms )
        
        return infos
    
    
    def get_frames_info(self):
        box:List[np.ndarray] = []
        coord:List[np.ndarray] = []
        types:List[np.ndarray] = []
        energy:List[np.ndarray] = []
        force:List[np.ndarray] = []
        if self.virial_mark:
            virial:List[np.ndarray] = []
        if self.eatoms_mark:
            eatoms:List[np.ndarray] = []
        if self.magmoms_mark:
            magmoms:List[np.ndarray] = []
        with open(self.mv_path, "r") as mvt:
            for idx_chunk in range(len(self.chunksizes)):
                tmp_chunk = ""
                for idx_chunk_line in range(self.chunksizes[idx_chunk]):
                    tmp_chunk += mvt.readline()
                ac_str_extractor = ACstrExtractor(atom_config_str=tmp_chunk)
                box.append( ac_str_extractor.get_basis_vectors() )
                coord.append( ac_str_extractor.get_coords() )
                types.append( ac_str_extractor.get_types() )
                energy.append( ac_str_extractor.get_etot() )
                force.append( ac_str_extractor.get_fatoms() )
                if self.virial_mark:
                    virial.append( ac_str_extractor.get_virial() )
                if self.eatoms_mark:
                    eatoms.append( ac_str_extractor.get_eatoms() )
                if self.magmoms_mark:
                    magmoms.append( ac_str_extractor.get_magmoms() )
        
        info:List[List[np.ndarray]] = []
        info.append( np.array(box) )
        info.append( np.array(types) )
        info.append( np.array(coord) )
        info.append( np.array(energy) )
        info.append( np.array(force) )
        if self.virial_mark:
            info.append( np.array(virial) )
        if self.eatoms_mark:
            info.append( np.array(eatoms) )
        if self.magmoms_mark:
            info.append( np.array(magmoms) )
        
        return info
            

    def get_info_labels(self):
        info_labels = []
        info_labels.append("box")
        info_labels.append("types")
        info_labels.append("coords")
        info_labels.append("etot")
        info_labels.append("fatoms")
        
        if self.virial_mark:
            info_labels.append("virial")
        if self.eatoms_mark:
            info_labels.append("etaoms")
        if self.magmoms_mark:
            info_labels.append("magmoms")
        
        return info_labels
    
    
    def _find_extra_properties(self):
        virial_mark:bool = True
        eatoms_mark:bool = True
        magmoms_mark:bool = True
        
        frame_0_str:str = self.get_frame_str(fidx=0)
        ac_str_extractor = ACstrExtractor(atom_config_str=frame_0_str)
        try:
            _ = ac_str_extractor.get_virial()
        except AttributeError as e:
            virial_mark = False
        
        try:
            _ = ac_str_extractor.get_eatoms()
        except AttributeError as e:
            eatoms_mark = False
        
        try:
            _ = ac_str_extractor.get_magmoms()
        except AttributeError as e:
            magmoms_mark = False
        
        
        return [virial_mark, eatoms_mark, magmoms_mark]
    