import numpy as np
import itertools

from ...io.publicLayer.structure import DStructure


class AdjacentMatrix(object):
    '''
    Description
    -----------
        1. 用于计算邻接矩阵(Adjacent Matrix)
    '''
    def __init__(self, 
                structure:DStructure,
                rcut:float,
                scaling_matrix:list
                ):
        '''
        Parameters
        ----------
            1. structure: DStructure
                - 结构
            2. rcut: float
                - 截断半径
            3. scaling_matrix: List[int]
                - 扩包倍数
                - Note: 只能是奇数
        '''
        self.structure = structure
        self.rcut = rcut
        self.scaling_matrix = scaling_matrix
        
        # 确保扩包倍数为奇数
        assert ( (self.scaling_matrix[0] > 0) and (self.scaling_matrix[0] / 2 != 0) )
        assert ( (self.scaling_matrix[1] > 0) and (self.scaling_matrix[1] / 2 != 0) )
        assert ( (self.scaling_matrix[2] > 0) and (self.scaling_matrix[2] / 2 != 0) )
    
    
    def get_adjacent_matrix(self):
        '''
        Description
        -----------
            1. 得到邻接矩阵 (adjacent matrix)
        '''
        ### Step 1. 得到原子总数和 primitive_cell 的向量基矢
        num_atoms = self.structure.num_sites
        basis_vectors_array = self.structure.lattice.matrix
        
        ### Step 2. 扩包后，周围会有很多primitive_cell，需要得到新primitive_cell 的 `分数原子坐标`
        ###     Note: 包含了本身的 primitive_cell
        # shape = (num_atoms_in_primitive_cell, 3, 1)
        primitive_frac_coords = np.repeat(
                                    self.structure.frac_coords[:, :, np.newaxis],
                                    1,
                                    axis=-1)
        # shape = (num_atoms_in_primitive_cell, 3, 27)
        shifted_supercell_frac_coords = self._get_shifted_supercell_frac_coords()
        
        ### Step 3. 初始化 adjacent_matrix
        adjacent_matrix = np.zeros((num_atoms, num_atoms))
        
        ### Step 4. 根据上述信息获取 adjacent matrix
        for idx_atom_1 in range(num_atoms):
            for idx_atom_2 in range(idx_atom_1, num_atoms):
                    ### Step 4.1. 
                # shape = (3, 27)
                atom_frac_diff = shifted_supercell_frac_coords[idx_atom_2] - primitive_frac_coords[idx_atom_1]
                distance_ij = np.dot(basis_vectors_array.T, atom_frac_diff)
                if sum(np.linalg.norm(distance_ij, axis=0) <= self.rcut) > 0:
                    adjacent_matrix[idx_atom_1, idx_atom_2] = sum(np.linalg.norm(distance_ij, axis=0) <= self.rcut)
                    adjacent_matrix[idx_atom_2, idx_atom_1] = sum(np.linalg.norm(distance_ij, axis=0) <= self.rcut)
        
        ### Step 5. 减去该原子与自身的近邻的情况
        adjacent_matrix = adjacent_matrix - np.eye(num_atoms)
        
        return adjacent_matrix
        
        
        
    
    def _get_shifted_supercell_frac_coords(self):
        '''
        Description
        -----------
            1. 扩包后，周围会有很多primitive_cell，需要得到新primitive_cell 的 `分数原子坐标`
            2. Note: 包含本身的 primitive_cell
            
        Return
        ------
            1. neigh_primitive_cell_frac_coords: np.ndarray
                - supercell 中所有原子距 primitive_cell 原子的分数坐标
                - shape = (12, 3, 27)
                    - `12`: 12 个原子
                    - `3` : x, y, z 三个坐标
                    - `27`: 3 * 3 * 3 (扩包倍数)
        '''
        ### Step 1. 获取扩胞前primitive_cell的 `原子分数坐标`
        frac_coords = np.array(self.structure.frac_coords)
        
        ### Step 2. 扩包时，各个primitive_cell相对于未扩包的primitive_cell的移动
        '''
        frac_shift_matrix 
        -----------------
            - e.g. self.scaling_matrix = [3, 3, 3]
                [[-1 -1  0]
                [-1  0  0]
                [-1  1  0]
                [ 0 -1  0]
                [ 0  0  0]
                [ 0  1  0]
                [ 1 -1  0]
                [ 1  0  0]
                [ 1  1  0]]
                ...
                ...
        '''
        shift_x = int( (self.scaling_matrix[0]-1) / 2 )
        shift_y = int( (self.scaling_matrix[1]-1) / 2 )
        shift_z = int( (self.scaling_matrix[2]-1) / 2 )
        # 转置后 shape = (3, 27)
        frac_shift_matrix_ = np.array(
                            list(itertools.product(
                                        list(range(-shift_x, shift_x+1)),
                                        list(range(-shift_y, shift_y+1)),
                                        list(range(-shift_z, shift_z+1))
                            ))
        ).T
        
        ### Step 3. `neigh_primitive_cell_frac_coords`
        # shape = (12, 3, 27)
        supercell_frac_coords = np.repeat(
                                    frac_coords[:, :, np.newaxis],
                                    self.scaling_matrix[0] * self.scaling_matrix[1] * self.scaling_matrix[2],
                                    axis=2)
        # shape = (1, 3, 27)
        frac_shift_matrix = np.repeat(
                                    frac_shift_matrix_[np.newaxis, :, :],
                                    1,
                                    axis=0)
        # shape = (12, 3, 27)
        shifted_supercell_frac_coords = supercell_frac_coords - frac_shift_matrix
        
        return shifted_supercell_frac_coords