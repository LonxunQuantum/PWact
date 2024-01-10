import numpy as np


class VirialTensor(object):
    def __init__(self, relative_coords:np.ndarray, f_atoms:np.ndarray):
        self.relative_coords = relative_coords
        self.f_atoms = f_atoms
        
    
    def calculate_virial_value(self, direction_1:int, direction_2:int):
        