import linecache

import numpy as np


class OutFermi(object):
    def __init__(self,
            out_fermi_path:str):
        self.out_fermi_path = out_fermi_path
    
    
    def get_efermi(self):
        '''
        Description
        -----------
            1. 从 `self.out_fermi_path` 中读取费米能级(unit: eV)
        '''
        first_row = linecache.getline(self.out_fermi_path, 1)
        first_row_lst = first_row.split()
        efermi_ev = np.round(float(first_row_lst[-2]), 3)
        return efermi_ev