import os 
import warnings
import numpy as np
import copy
from typing import List, Dict
import multiprocessing as mp

from ..utils.lineLocator import (LineLocator,
                                 ListLocator)
from ..utils.parameters import atomic_number2specie
from ...publicLayer.structure import DStructure
from ...publicLayer.traj import Trajectory
from ...publicLayer.neigh import StructureNeighborsUtils


class Movement(Trajectory):
    def __init__(self,
                movement_path:str
                ):
        '''
        Parameters
        ----------
            1. movement_path: str
                - MOVEMENT 文件的路径
            2. 
            
        Note
        ----
            1. MOVEMENT 第一步与其他步的 chunksize 不同。
                - chunksize 值得是每个 frame 在 MOVEMENT 中占用的行数
        '''
        self.movement_path = movement_path
        self.chunksizes_lst = self.get_chunksize()
        self.chunkslices_lst = self.get_chunkslice()
        
        
    def get_chunksize(self,
                    ):
        '''
        Description
        -----------
            1. 由于MOVEMENT 文件太大，因此将 MOVEMENT 的每一帧 (frame) 对应的内容，定义为一个chunk
            文件处理时 `一个chunk一个chunk处理`
            
        Return
        ------
            1. chunk_size: int 
                - 每一帧的行数
        
        Note
        ----
            1. chunksize 包括 `-------` 这一行
        '''
        chunksizes_lst = []
        content = "--------------------------------------"
        row_idxs = LineLocator.locate_all_lines(
                                    file_path=self.movement_path,
                                    content=content
                                    )

        chunksizes_lst.append(row_idxs[0])
        for idx in range(1, len(row_idxs)):
            chunksizes_lst.append(row_idxs[idx] - row_idxs[idx-1])
        
        return chunksizes_lst
    
    
    def get_chunkslice(self):
        '''
        Description
        -----------
            1. 
        
        Return
        ------
            1. chunslice: List[int]
                - e.g. `[     0    225    607    989   1371 ...]`
                - chunkslice[idx_frame: idx_frame+1] -- 某个frame的起始行: 终止行
        '''
        chunksizes_lst = copy.deepcopy(self.chunksizes_lst)
        chunksizes_lst.insert(0, 0)    
        chunkslice = np.cumsum(chunksizes_lst)
        
        return chunkslice

    
    def _get_frame_str(self, idx_frame:int):
        '''
        Description
        -----------
            1. 得到某一帧的str
            2. Note: 帧数从 0 开始
        
        Parameter
        ---------
            1. idx_frame: int
                - 得到代表某一帧的 str
            
        Return
        ------
            1. str_frame: str
                - 某一帧的 str
        '''      
        ### Step 1. 
        tmp_chunk = ""
        with open(self.movement_path, "r") as mvt:
            for idx_line, line in enumerate(mvt):
                #print(idx_line, self.chunkslices_lst[idx_frame], self.chunkslices_lst[idx_frame+1])
                if idx_line in range(self.chunkslices_lst[idx_frame], self.chunkslices_lst[idx_frame+1]):
                    tmp_chunk += line
                elif idx_line >= self.chunkslices_lst[idx_frame+1]:
                    break
                
        return tmp_chunk
            
    
    def get_all_frame_structures(self):
        '''
        Description
        -----------
            1. 返回 Movement 中的所有结构: List[DStructure]
        '''
        structures_lst = []
        
        with open(self.movement_path, "r") as mvt:
            for idx_chunk in range( len(self.chunksizes_lst) ):
                tmp_chunk = ""
                for idx_line in range(self.chunksizes_lst[idx_chunk]):
                    tmp_chunk += mvt.readline()
                ### Step 1. 得到 DStructure object
                structures_lst.append(
                        DStructure.from_str(
                                str_content=tmp_chunk,
                                str_format="pwmat")
                )
                
        return structures_lst
    
    
    def get_all_frame_structures_info(self):
        '''
        Description
        -----------
            1. 返回 Movement 中的所有:
                1. 结构(DStructure)
                2. 能量 (总能、动能、势能)
                3. virial tensor
                
        Return
        ------
            1. structures_lst: List[DStructure]
                - (num_frames,)
            2. np.array(total_energys_lst) : np.array
                - (num_frames,)
            3. np.array(potential_energys_lst) : np.array
                - (num_frames,)
            4. np.array(kinetic_energys_lst) : np.array 
                - (num_frames,)
            5. np.array(virial_tensors_lst) : np.array
                - (num_frames, 3, 3)
        '''
        structures_lst = []
        total_energys_lst = []
        potential_energys_lst = []
        kinetic_energys_lst = []
        virial_tensors_lst = []
        
        with open(self.movement_path, "r") as mvt:
            for idx_chunk in range(len(self.chunksizes_lst)):
                tmp_chunk = ""
                for idx_line in range(self.chunksizes_lst[idx_chunk]):
                    tmp_chunk += mvt.readline()
                ### Step 1. 得到 DStructure object
                structures_lst.append(
                        DStructure.from_str(
                                    str_content=tmp_chunk, 
                                    str_format="pwmat")
                )
                
                ### Step 2. Energy (Etot, Ep, Ek)
                first_row_lst = tmp_chunk.split('\n')[0].split()
                # 第二个等号前面是Etot,Ep,Ek，后面三个数分别是 Etot, Ep, Ek
                equal_1_idx = ListLocator.locate_all_lines(strs_lst=first_row_lst, content="=")[1]
                energy_tot = float( first_row_lst[equal_1_idx+1].strip() )
                energy_p = float( first_row_lst[equal_1_idx+2].strip() )
                # 54 atoms,Iteration=    0.0000000000E+00, Etot,Ep,Ek=   -0.1062748168E+05  -0.1062748168E+05   0.0000000000E+00
                # 72 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1188642969E+05  -0.1188642969E+05   0.0000000000E+00, SCF =    16
                energy_k = first_row_lst[equal_1_idx+3].strip()
                if energy_k[-1] == ',':
                    energy_k = float(energy_k[:-1])
                else:
                    energy_k = float(energy_k)
                
                total_energys_lst.append(energy_tot)
                potential_energys_lst.append(energy_p)
                kinetic_energys_lst.append(energy_k)
                
                ### Step 4. Virial Tensor
                chunk_rows_lst = tmp_chunk.split("\n")
                num_atoms = int(chunk_rows_lst[0].split()[0])   # 每个 frame 的原子数目
                aim_idx = ListLocator.locate_all_lines(strs_lst=chunk_rows_lst,
                                                    content="LATTICE VECTOR")[0]
                
                if len(chunk_rows_lst[aim_idx+1].split()) == 3:
                    pass
                else:
                    virial_tensor_x = np.array([float(tmp_value.strip()) for tmp_value in chunk_rows_lst[aim_idx+1].split()[-3:]])
                    virial_tensor_y = np.array([float(tmp_value.strip()) for tmp_value in chunk_rows_lst[aim_idx+2].split()[-3:]])
                    virial_tensor_z = np.array([float(tmp_value.strip()) for tmp_value in chunk_rows_lst[aim_idx+3].split()[-3:]])
                    virial_tensor = np.vstack([virial_tensor_x, virial_tensor_y, virial_tensor_z])
                
                    virial_tensors_lst.append(virial_tensor)
        
        return (
                structures_lst,
                np.array(total_energys_lst),
                np.array(potential_energys_lst),
                np.array(kinetic_energys_lst),
                np.array(virial_tensors_lst)    # 若没有virial tensor信息，则为np.array([])
        )
                    
    
    def get_frame_structure(self, idx_frame:int):
        '''
        Description
        -----------
            1. 将某一帧的结构取出来，构建成 DStructure 对象
        
        Parameters
        ----------
            1. idx_frame: int
                - 第几帧 (从 0 开始计数)
        
        Return
        ------
            1. structure: DStructure
                - 
        '''
        str_frame = self._get_frame_str(idx_frame=idx_frame)
        structure = None
        structure = DStructure.from_str(
                                str_content=str_frame,
                                str_format="pwmat",
                                coords_are_cartesian=False)
        return structure
    
    
    def get_frame_energy(self, idx_frame:int):
        '''
        Description
        -----------
            1. 获取某一帧的总能、势能、动能
                72 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1188642969E+05  -0.1188642969E+05   0.0000000000E+00, SCF =    16
                Lattice vector (Angstrom)
                0.8759519000E+01    0.0000000000E+00    0.0000000000E+00     stress (eV):  0.124196E+02  0.479262E+01  0.245741E+01
                0.2209000000E+00    0.7513335000E+01    0.0000000000E+00     stress (eV):  0.479308E+01  0.961132E+01  0.225365E+01
                0.4093050000E+00    0.2651660000E+00    0.1828974400E+02     stress (eV):  0.245631E+01  0.225430E+01 -0.198978E+01
        
        Paramters
        ---------
            1. idx_frame: int
                - 第几帧 (从第 0 帧开始计数)
        
        Return
        ------
            1. energy_tot: float
            2. energy_p: float
            3. energy_k: float
        '''
        ### 1. 获取 `idx_frame` 对应的 chunk
        frame_str = self._get_frame_str(idx_frame=idx_frame)

        ### 2. 获取Etot, Ep, Ek
        first_row_lst = frame_str.split('\n')[0].split()    # 某一帧 chunk 的第一行
        energy_tot = float( first_row_lst[8].strip() )
        energy_p = float( first_row_lst[9].strip() )
        energy_k = float( first_row_lst[10][:-1].strip() )

        return energy_tot, energy_p, energy_k
    
    
    def get_frame_virial(self, idx_frame:int):
        '''
        Description
        -----------
            1. 获取某一帧的维里张量
                72 atoms,Iteration (fs) =   -0.1000000000E+01, Etot,Ep,Ek (eV) =   -0.1188642969E+05  -0.1188642969E+05   0.0000000000E+00, SCF =    16
                Lattice vector (Angstrom)
                0.8759519000E+01    0.0000000000E+00    0.0000000000E+00     stress (eV):  0.124196E+02  0.479262E+01  0.245741E+01
                0.2209000000E+00    0.7513335000E+01    0.0000000000E+00     stress (eV):  0.479308E+01  0.961132E+01  0.225365E+01
                0.4093050000E+00    0.2651660000E+00    0.1828974400E+02     stress (eV):  0.245631E+01  0.225430E+01 -0.198978E+01
        
        Parameters
        ----------
            1. idx_frame: int
                - 第几帧 (从第 0 帧开始计数)
        
        Return
        ------
            1. virial_tensor: np.ndarray
        '''
        ### 1. 获取 `idx_frame` 对应的 chunk，并以 `\n` 为分裂标志将其分成列表
        frame_str = self._get_frame_str(idx_frame=idx_frame)
        rows_lst = frame_str.split("\n")

        ### 2. 找到 Lattice vector 对应的行的索引 -- `aim_idx`
        aim_content = "LATTICE VECTOR"
        aim_idx = ListLocator.locate_all_lines(strs_lst=rows_lst, content=aim_content)[0]
        
        ### 3. 将 virial tensor 转换成 3*3 的 np.ndarray
        if ( len(rows_lst[aim_idx+1].split()) == 3 ):   # 没有virial信息，只有lattice信息
            virial_tensor = np.zeros((3, 3))
        else:
            virial_tensor_x = np.array([float(tmp_value.strip()) for tmp_value in rows_lst[aim_idx+1].split()[-3:]])
            virial_tensor_y = np.array([float(tmp_value.strip()) for tmp_value in rows_lst[aim_idx+2].split()[-3:]])
            virial_tensor_z = np.array([float(tmp_value.strip()) for tmp_value in rows_lst[aim_idx+3].split()[-3:]])
            virial_tensor = np.vstack([virial_tensor_x, virial_tensor_y, virial_tensor_z])
        
        return virial_tensor
            
    