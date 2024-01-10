from abc import ABC, abstractmethod


class Trajectory(ABC):
    '''
    Description
    -----------
        1. A file record the trajectory of AIMD
    '''
    @abstractmethod
    def get_chunksize(self):
        pass
    
    
    @abstractmethod
    def get_chunkslice(self):
        pass
    
    @abstractmethod
    def _get_frame_str(self, idx_frame:int):
        pass
    
    
    @abstractmethod
    def get_all_frame_structures(self):
        pass
    
    
    @abstractmethod
    def get_all_frame_structures_info(self):
        pass
    
    
    @abstractmethod
    def get_frame_structure(self, idx_frame:int):
        pass
    
    
    @abstractmethod
    def get_frame_energy(self, idx_frame:int):
        pass
    
    
    @abstractmethod
    def get_frame_virial(self, idx_frame:int):
        pass
    
    
    @abstractmethod
    def get_frame_volume(self, idx_frame:int):
        pass
    
    
    @abstractmethod
    def get_max_num_nbrs_real(self):
        pass