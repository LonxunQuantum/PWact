'''
1. ACExtractor: Extract info from atom.config

2. ACEstrExtractor: Extract info from string of MOVEMENT
'''
import re
import linecache
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from .lineLocator import LineLocator, ListLocator
from .parameters import atomic_number2specie


class ACExtractorBase(ABC):
    @abstractmethod
    def get_num_atoms(self):
        pass
    
    @abstractmethod
    def get_basis_vectors(self):
        pass
    
    @abstractmethod
    def get_types(self):
        pass
    
    @abstractmethod
    def get_coords(self):
        '''
        Description
        -----------
            1. 提取体系的分数坐标
        '''
        pass

    def get_magmoms(self):
        pass
    
    def get_eatoms(self):
        pass

    def get_etot(self):
        pass
    
    def get_fatoms(self):
        pass
    
    @abstractmethod
    def get_magmoms(self):
        pass
    
        
class ACExtractor(ACExtractorBase):
    '''
    Description
    -----------
        1. 提取 atom.config 文件中的信息：
            1. num_atoms: int
                体系内的原子总数
            2. basis_vectors: list
                体系的基矢 (二维list)
            3. types: list of str
                各个 site 的原子种类
            4. coords: np.array
                各个 site 的分数坐标
            5. 
    
        2. 当提取的是 MOVEMENT 中某一帧信息时，还可以提取到：
            1. `Force (eV/Angstrom)`
            2. `Atomic-Energy`
            3. ...
    '''
    def __init__(self,
                file_path: str):
        '''
        Parameters
        ----------
            1. atom_config_path: str
                The absolute path of `atom.config` file
        '''
        self.atom_config_path = file_path
        self.num_atoms = self.get_num_atoms()
        self.basis_vectors = self.get_basis_vectors()
        self.types = self.get_types()
        self.coords = self.get_coords()
        #self.magnetic_moments = self.get_magnetic_moments()


    def get_num_atoms(self):
        '''
        Description
        -----------
            1. 得到体系的总原子数目
        '''
        # atom.config 文件第一行综合描述这个体系
        first_row = linecache.getline(self.atom_config_path, 1)
        num_atoms = int( first_row.split()[0] )
        return num_atoms


    def get_basis_vectors(self):
        '''
        Description
        -----------
            1. 得到材料的基矢

        Return
        ------
            1. basis_vectors: np.ndarray (shape = (9,))
            e.g. [13.13868  0.       0.       0.      13.13868  0.       0.       0.    
                  13.13868]
            
        '''
        basis_vectors:List[float] = []

        ### Step 1. 得到所有原子的原子序数、坐标
        content = "LATTICE"    # 此处需要大写
        idx_row = LineLocator.locate_all_lines(
                            file_path=self.atom_config_path,
                            content=content)[0]

        ### Step 2. 获取基矢向量
        for row_idx in [idx_row+1, idx_row+2, idx_row+3]:
            row_content:List[str] = linecache.getline(self.atom_config_path, row_idx).split()[:3]
            
            for value in row_content:
                basis_vectors.append(float(value))
                
        return np.array(basis_vectors)
        
    
    def get_types(self):
        '''
        Description
        -----------
            1. 得到体系内所有的原子序数 (各个原子的 atomic_numbers)
            
        Return
        ------
            1. atomic_numbers : np.ndarray

        Note
        ----
            1. 重复
        '''
        ### Part I. 得到所有原子的原子序数、坐标
        ###         所有原子的原子序数：atomic_numbers_lst
        ###         所有原子的坐标：coordinations_lst
        content = "POSITION"    # 此处需要大写
        idx_row = LineLocator.locate_all_lines(
                                file_path=self.atom_config_path,
                                content=content)[0]
        with open(self.atom_config_path, 'r') as f:
            atom_config_content = f.readlines()
        
        # 1. 得到所有原子的原子序数（注意将读取的 str 转换为 int）
        atomic_numbers_content = atom_config_content[idx_row:idx_row + self.num_atoms]
        atomic_numbers_lst = [int(row.split()[0]) for row in atomic_numbers_content]

        return np.array(atomic_numbers_lst)


    def get_coords(self):
        '''
        Description
        -----------
            1. 得到体系内所有的分数坐标
            
        Return
        ------
            1. coords : np.ndarray
                - shape = (num_atoms * 3,)
        '''
        coords_lst = []
        content = "POSITION"    # 此处需要大写
        idx_row = LineLocator.locate_all_lines(
                                    file_path=self.atom_config_path,
                                    content=content)[0]
        with open(self.atom_config_path, 'r') as f:
            atom_config_content = f.readlines()
        
        # 1. 得到所有原子的原子序数（注意将读取的 str 转换为 int）
        """
        row_content
        -----------
            '29         0.377262291145329         0.128590184800933         0.257759805813488     1  1  1'
        """
        for row_content in atom_config_content[idx_row:idx_row + self.num_atoms]:
            row_content_lst = row_content.split()
            coord_tmp = [float(value) for value in row_content_lst[1:4]]
            coords_lst.append(np.array(coord_tmp))
        
        return np.array(coords_lst).reshape(-1)
        

    def get_magmoms(self):
        '''
        Description
        -----------
            1. 得到所有原子的磁矩，顺序与 `坐标` 的顺序一致
        '''
        content = "MAGNETIC"

        magnetic_moments_lst = []
        
        try:    # 处理异常：若 atom.config 中不包含原子的磁矩信息
            idx_row = LineLocator.locate_all_lines(
                                    file_path=self.atom_config_path,
                                    content=content)[-1]

            with open(self.atom_config_path, "r") as f:
                atom_config_content = f.readlines()
            
            magnetic_moments_content = atom_config_content[idx_row: idx_row+self.num_atoms]
            # MAGNETIC  
            # 3 0.0 # 原子序数 磁矩
            # ...
            magnetic_moments_lst = [float(tmp_magnetic_moment.split()[-1]) for tmp_magnetic_moment in magnetic_moments_content]
        except Exception as e:
            #print(e)
            magnetic_moments_lst = [0 for _ in range(self.num_atoms)]
        
        return magnetic_moments_lst
    
    
class ACstrExtractor(ACExtractorBase):
    '''
    Description
    -----------
        1. 从 str 中提取 atom.config 的信息
        2. Note: str 一般是从 MOVEMENT 中切片获取
    '''
    def __init__(self, atom_config_str:str):
        self.atom_config_str = atom_config_str
        self.strs_lst = self.atom_config_str.split('\n')
        
        self.num_atoms = self.get_num_atoms()
        #self.basis_vectors_array = self.get_basis_vectors()
        #self.species_array = [atomic_number2specie[atomic_number] for atomic_number in self.get_atomic_numbers_lst() ]
        #self.coords_array = self.get_coords_lst()
        #self.magnetic_moments = self.get_magnetic_moments()
        
    
    def get_num_atoms(self):
        match_object = re.search(r"(\d+) atoms", self.atom_config_str, re.IGNORECASE)
        num_atoms = int(match_object.group(1))
        return num_atoms
    
    
    def get_basis_vectors(self):
        '''
        Description
        -----------
            1. 得到材料的基矢

        Return
        ------
            1. basis_vectors: np.ndarray
            e.g. [13.13867997  0.          0.          0.         13.13867997  0.
                  0.          0.         13.13867997]

        '''
        basis_vectors_lst = []
        ### Step 1. 得到 `LATTICE` 在列表中的索引
        aim_content = "LATTICE"
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]

        ### Step 2. 
        for idx_str in [aim_idx+1, aim_idx+2, aim_idx+3]:
            # ['0.8759519000E+01', '0.0000000000E+00', '0.0000000000E+00']
            str_lst = self.strs_lst[idx_str].split()[:3]
            for tmp_str in str_lst:
                basis_vectors_lst.append(float(tmp_str))

        return np.array(basis_vectors_lst)
        

    def get_types(self):
        '''
        Description
        -----------
            1. 得到体系内所有的原子序数 (各个原子的 atomic_numbers)

        Note
        ----
            1. 重复
        '''
        ### Step 1. 获取 `POSITION` 所在的行数
        aim_content = "POSITION"
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]

        ### Step 2. 得到所有原子的原子序数（注意将 str 转换为 int）        
        strs_lst = self.strs_lst[aim_idx+1 : aim_idx + self.num_atoms + 1]
        atomic_numbers_lst = [int(entry.split()[0]) for entry in strs_lst]
        
        return np.array(atomic_numbers_lst)
    
    
    def get_coords(self):
        '''
        Description
        -----------
            1. 得到体系内所有的分数坐标
        '''
        coords_lst = []
        aim_content = "POSITION"    # 此处需要大写
        aim_idx = ListLocator.locate_all_lines(
                                strs_lst=self.strs_lst,
                                content=aim_content)[0]

        for tmp_str in self.strs_lst[aim_idx+1: aim_idx+self.num_atoms+1]:
            # ['14', '0.751401861790384', '0.501653718883189', '0.938307102003243', '1', '1', '1']
            tmp_strs_lst = tmp_str.split()
            tmp_coord = [float(value) for value in tmp_strs_lst[1:4]]
            coords_lst.append(tmp_coord)
        
        return np.array(coords_lst).reshape(-1)
    
    
    def get_magmoms(self):
        '''
        Description
        -----------
            1. 得到所有原子的磁矩，顺序与 `坐标` 的顺序一致
        '''
        magnetic_moments_lst = []
        try:
            aim_content = "MAGNETIC"
            aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)
            
            magnetic_moments_content = self.strs_lst[aim_idx+1: aim_idx+self.num_atoms+1]
            magnetic_moments_lst = [float(tmp_magnetic_moment.split()[-1]) for tmp_magnetic_moment in magnetic_moments_content]

            return np.array(magnetic_moments_lst)
        except:
            raise AttributeError("ACstrExtractor: No magmoms info in MOVEMENT.")
    
  
    def get_etot(self):
        '''
        [' 216 atoms', 'Iteration (fs) =    0.3000000000E+01', ' Etot', 'Ep', 'Ek (eV) =   -0.2831881714E+05  -0.2836665392E+05   0.4783678177E+02', ' SCF =     7']
        '''
        strs_lst = self.strs_lst[0].split(",")
        aim_index = ListLocator.locate_all_lines(strs_lst=strs_lst, content="EK (EV) =")[0]
        # strs_lst[aim_index].split() = ['Ek', '(eV)', '=', '-0.2831881714E+05', '-0.2836665392E+05', '0.4783678177E+02']
        return np.array([ float(strs_lst[aim_index].split()[3].strip()) ])
  
    
    def get_eatoms(self):
        '''
        Description
        -----------
            1. 得到体系内所有原子的能量
        '''
        try:
            eatoms_lst = []
            aim_content = "Atomic-Energy, ".upper()
            aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
            
            for tmp_str in self.strs_lst[aim_idx+1: aim_idx+self.num_atoms+1]:
                '''
                Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  -0.1281163115E+06
                14   0.6022241483E+03    0.2413350871E+02    0.3710442365E+01
                '''
                eatoms_lst.append(float( tmp_str.split()[1] ))
            return np.array(eatoms_lst)
        
        except:
            raise AttributeError("ACExtractorError: No atomic energy info in MOVMENT.")
  
  
    def get_fatoms(self):
        '''
        Description
        -----------
            1. 得到体系内所有原子的受力
        '''
        try:
            forces_lst = []
            aim_content = "Force".upper()
            aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
            
            for tmp_str in self.strs_lst[aim_idx+1: aim_idx+self.num_atoms+1]:
                # ['14', '0.089910342901203', '0.077164252174742', '0.254144099204679']
                tmp_str_lst = tmp_str.split()
                tmp_forces = [float(value) for value in tmp_str_lst[1:4]]
                forces_lst.append(tmp_forces)
            return np.array(forces_lst).reshape(-1)
        except: # atom_config_str 中没有关于原子受力的信息
            return np.zeros((self.num_atoms*3, ))
  
    
    def get_virial(self):
        '''
        Description
        -----------
            1. 得到材料的维里张量 (virial tensor)

        Return
        ------
            1. virial_tensor: np.array, 是一个二维 np.ndarray            
        '''
        virial_tensor = []
        
        ### Step 1. 得到所有原子的原子序数、坐标
        aim_content = "LATTICE"
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        
        for tmp_idx in [aim_idx+1, aim_idx+2, aim_idx+3]:
            # tmp_strs_lst = ['0.8759519000E+01', '0.0000000000E+00', '0.0000000000E+00', 'stress', '(eV):', '0.115558E+02', '0.488108E+01', '0.238778E+01']
            tmp_strs_lst = self.strs_lst[tmp_idx].split()
            tmp_aim_row_lst = ListLocator.locate_all_lines(strs_lst=tmp_strs_lst, content="STRESS")
            if len(tmp_aim_row_lst) == 0:
                raise AttributeError("ACstrExtractorError: No virial info in MOVEMENT.");
        
        ### Step 2. 获取维里张量
        for tmp_idx in [aim_idx+1, aim_idx+2, aim_idx+3]:
            # tmp_str_lst = ['0.120972E+02', '0.483925E+01', '0.242063E+01']
            tmp_str_lst = self.strs_lst[tmp_idx].split()[-3:]

            virial_tensor.append(float(tmp_str_lst[0]))
            virial_tensor.append(float(tmp_str_lst[1]))
            virial_tensor.append(float(tmp_str_lst[2]))
        
        return np.array(virial_tensor)

    