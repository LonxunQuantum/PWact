import os
import numpy as np
from typing import List
import multiprocessing as mp

from ...io.publicLayer.structure import DStructure
from ...io.publicLayer.traj import Trajectory


class DiffractionIntensity(object):
    '''
    Description
    -----------
        1. We calculate the `time-dependent diffraction intensity` based on the atomic trajectories produced by 
        the rt-TDDFT simulations according to the Debye-Waller factor.
    '''
    def __init__(self, trajectory:Trajectory, q:float):
        '''
        Description
        -----------
            1. 
        
        Parameters
        ----------
            1. trajectory: Trajectory
                - 轨迹对象
            2. q: float
                - q is the magnitude of the reciprocal lattice vector for the diffraction spot.
        '''
        self.q = q
        self.msd_array = np.array( Msd(trajectory=trajectory).calc_msd() )
    
    
    def calc_di(self):
        '''
        Description
        -----------
            1. 
        '''
        di_array = np.exp(-self.q**2 * self.msd_array / 3)
        return di_array
        
    


class Msd(object):
    '''
    Description
    -----------
        1. 根据轨迹文件计算 Mean Squared Displacement.
    '''
    def __init__(self, trajectory:Trajectory):
        self.structures_lst = trajectory.get_all_frame_structures()
        
    
    def calc_msd(self):
        '''
        Description
        -----------
            1. Parallel the `MsdParallelFunction.calc_msd_s(structure_1, structure_2)`
            2. Calculate `Mean squared displacement` -- `MSD`
                - MSD = 1/n sum_{i=1}^n [x_i(t) - x_origin]^2
            3. 不减去质心
        '''
        parameters_lst:List[List] = []
        for tmp_frame_idx in range(1, len(self.structures_lst)):
            parameters_lst.append([
                        self.structures_lst[tmp_frame_idx-1],
                        self.structures_lst[tmp_frame_idx]]
            )
        
        msd_values_lst:List[int] = []
        with mp.Pool(processes=os.cpu_count()-2) as pool:
            msd_values_lst = pool.starmap(
                                MsdParallelFunction.calc_msd_s,
                                parameters_lst
            )
        return msd_values_lst
    
    
    def calc_msd_sub_centroid(self):
        '''
        Description
        -----------
            1. Parallel the `MsdParallelFunction.calc_msd_s(structure_1, structure_2)`
            2. Calculate `Mean squared displacement` -- `MSD`
                - MSD = 1/n sum_{i=1}^n [ (x_i(t) - centroid(t)) - (x_origin - centroid_origin) ]^2
            3. 减去质心
        '''
        parameters_lst:List[List] = []
        for tmp_frame_idx in range(1, len(self.structures_lst)):
            parameters_lst.append([
                            self.structures_lst[tmp_frame_idx-1],
                            self.structures_lst[tmp_frame_idx]]
            )
        
        msd_values_lst:List[int] = []
        with mp.Pool(processes=os.cpu_count()-2) as pool:
            msd_values_lst = pool.starmap(
                                MsdParallelFunction.calc_msd_sub_centroid_s,
                                parameters_lst
            )
        
        return msd_values_lst
            
        
class MsdParallelFunction(object):
    @staticmethod
    def calc_msd_s(
                structure_1:DStructure,
                structure_2:DStructure):
        '''
        Description
        -----------
            1. Calculate `Mean squared displacement` -- `MSD`
                - MSD = 1/n sum_{i=1}^n [x_i(t) - x_origin]^2
        '''
        relative_coords = structure_2.cart_coords - structure_1.cart_coords
        num_atoms = relative_coords.shape[0]
        
        return np.sum(np.power(relative_coords, 2)) / num_atoms


    @staticmethod
    def calc_msd_sub_centroid_s(
                structure_1:DStructure,
                structure_2:DStructure):
        '''
        Description
        -----------
            1. Calculate `Mean squared displacement` -- `MSD`
                - MSD = 1/n sum_{i=1}^n [ (x_i(t) - centroid(t)) - (x_origin - centroid_origin) ]^2
        '''
        relative_coords = (structure_2.cart_coords - structure_2.get_centroid().reshape(1, 3)) - \
                        (structure_1.cart_coords - structure_1.get_centroid().reshape(1,3))
        num_atoms = relative_coords.shape[0]
        
        return np.sum(np.power(relative_coords, 2)) / num_atoms