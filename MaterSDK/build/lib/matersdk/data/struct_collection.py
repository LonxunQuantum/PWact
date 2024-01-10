import os
import shutil
import numpy as np
from typing import List, Dict
import multiprocessing as mp

from ..io.pwmat.utils.parameters import atomic_number2specie
from ..io.publicLayer.traj import Trajectory
from ..io.publicLayer.structure import DStructure
from ..io.publicLayer.neigh import (
                        StructureNeighborsDescriptor,
                        StructureNeighborsUtils
)


class StructCollection(object):
    '''
    Description
    -----------
        1. `DeepDataSystem` 中包含很多frame的DStructure对象 (所有frame的原子数相同!!!)
        2. 这个类用于....?
        
    Attributions
    ------------
        1. self.num_structures: int
        2. self.structures_lst: List[Dstructure]
        3. self.total_energys_lst: List[float]
        4. self.kinetic_energys_lst: List[float]
        5. self.potential_energys_lst: List[float]
        6. self.virial_tensors_lst: List[np.ndarray]
        7. self.atomic_numbers_lst: List[int]
        8. self.num_atoms
    '''
    def __init__(
                self,
                structures_lst: List[DStructure],
                total_energys_array: np.ndarray,
                potential_energys_array: np.ndarray,
                kinetic_energys_array: np.ndarray,
                virial_tensors_array: np.ndarray):        
        self.num_structures = len(structures_lst)
        self.structures_lst = structures_lst
        self.total_energys_array = total_energys_array
        self.potential_energys_array = potential_energys_array
        self.kinetic_energys_array = kinetic_energys_array
        self.virial_tensors_array = virial_tensors_array
        
        self.atomic_numbers_lst = self._get_atomic_numbers()    # 不重复
        self.an2na = self._get_num_atoms_per_element()  # {3: 48, 14: 24} ({"原子序数": "对应的原子个数"} )
        self.num_atoms = self._get_num_atoms()
    
    
    def __str__(self):
        return self.__repr__()
    
    
    def __repr__(self):
        # Frame Numbers:
        # Atom Numbers
        # Including Virials:
        # Element List
        print("{0:*^60s}".format(" StructCollection Summary "))
        
        print("\t * {0:<24s}: {1:<14d}".format("Images Number", self.num_structures))
        print("\t * {0:<24s}: {1:<14d}".format("Atoms Number", self.num_atoms))
        print("\t * {0:<24s}: {1:<14}".format("Virials Information", f"{np.any(self.virial_tensors_array != 0)}"))
        mark_atomic_energy = np.any(self.structures_lst[0].sites[0].atomic_energy != 0)
        print("\t * {0:<24s}: {1:<14}".format("Energy Deposition", f"{mark_atomic_energy}"))
        print("\t * {0:<24s}:".format("Elements List"))
        for tmp_an in self.atomic_numbers_lst:
            print("\t\t - {0:<2s}: {1:<16d}".format(
                            atomic_number2specie[tmp_an], 
                            self.an2na[tmp_an]
                    )
            )
        
        print("{0:*^60s}".format("**"))
        return ""
    

    def __getitem__(self, index:int):
        return_structures_lst = self.structures_lst[index]
        return_total_energys_array = self.total_energys_array[index]
        return_potential_energys_array = self.potential_energys_array[index]
        return_kinetic_energys_array = self.kinetic_energys_array[index]
        return_virial_tensors_array = self.virial_tensors_array[index]
                
        return_object = StructCollection(
                            structures_lst=return_structures_lst,
                            total_energys_array=return_total_energys_array,
                            potential_energys_array=return_potential_energys_array,
                            kinetic_energys_array=return_kinetic_energys_array,
                            virial_tensors_array=return_virial_tensors_array,
        )

        return return_object
    
    
    def __len__(self):
        return self.num_structures
    
    
    def _get_num_atoms(self):
        '''
        Description
        -----------
            1. 得到每个system中原子的数目
        '''
        num_atoms = self.structures_lst[0].num_sites
        return num_atoms


    def _get_num_atoms_per_element(self):
        '''
        Description
        -----------
            1. 得到每个system中不同元素的原子数目
        
        Return
        ------  
            1. an2na: Dict[int, int]
                - e.g. {3: 48, 14: 24}
        '''
        structure = self.structures_lst[0]
        # an: atomic number
        # na: number of atoms
        an2na = {}
        for tmp_an in self.atomic_numbers_lst:
            tmp_na = [tmp_site.specie.Z for tmp_site in structure.sites if (tmp_site.specie.Z == tmp_an)]
            an2na.update({tmp_an: len(tmp_na)})
        return an2na
    
    
    def _get_atomic_numbers(self):
        '''
        Description
        -----------
            1. 得到体系内的原子序数
                - e.g. [3, 14]
        '''
        atomic_numbers_lst = [tmp_specie.Z for tmp_specie in self.structures_lst[0].species]
        atomic_numbers_lst = list(dict.fromkeys(atomic_numbers_lst))
        return atomic_numbers_lst
    
    
    @staticmethod
    def from_trajectory_s(trajectory_object:Trajectory):
        '''
        Description
        -----------
            1. 串行提取信息
        
        Parameters
        ----------
            1. trajectory_object: Trajectory
                - 轨迹对象
                - e.g. `matersdk.io.pwmat.output.movement.Movement`
        '''         
        ### Step 1. 得到 Movement 中所有构型的相关信息
        (structures_lst, 
        total_energys_array,
        potential_energys_array,
        kinetic_energys_array,
        virial_tensors_array) = \
                trajectory_object.get_all_frame_structures_info()
        
        
        ### Step 2. 初始化
        dp_data_system = StructCollection(
                        structures_lst=structures_lst,
                        total_energys_array=total_energys_array,
                        potential_energys_array=potential_energys_array,
                        kinetic_energys_array=kinetic_energys_array,
                        virial_tensors_array=virial_tensors_array,
        )
        
        return dp_data_system
    
    
    def to(
            self,
            dir_path:str,
            set_size:int):
        '''
        Description
        -----------
            1. 仅存储:
                1) box.npy
                2) coord.npy
                3) energy.npy
                4) force.npy
                5) virial.npy
        
        Parameters
        ----------
            1. dir_path: str
                - 输出的文件夹位置
            2. set_size: int
                - `set.xxx` 内 frame 的数量
        
        Note
        ----
            1. 与 dpdata 的输出格式一致
        '''
        ### Step 0. 如何 `dir_path文件夹` 不存在的话，则建立该文件夹
        if not os.path.exists(dir_path) :
            os.makedirs(dir_path)
            print(f"Folder created: {dir_path}")
        else:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            print(f"Step 1. Folder already exists: {dir_path}\nStep 2. Remove previous folder.\nStep 3. Create this folder again: {dir_path}") 
        
        ### Step 1. 存储:
        #       1. type_map.raw
        #       2. type.raw
        #       3. set.000
        ### Step 1.1. 存储 type_map.raw. e.g. "Li\nSi\n"
        # 根据 self.atomic_numbers_lst
        an2tm = {}   # atomic_number -> type_map {3: 0, 14: 1}
        for tmp_tm, tmp_an in enumerate(self.atomic_numbers_lst):
            an2tm.update({tmp_an: tmp_tm})

        type_map_str = "\n".join([str(tmp_tm) for tmp_tm in an2tm.values()])
        type_map_str += "\n"
        with open(f"{dir_path}/type_map.raw", 'w') as f:
            f.write(type_map_str)
        
        ### Step 1.2. 存储 type.raw e.g. "3\n3\n...14\n14\n..."
        type_lst = []
        for tmp_site_idx in range(self.num_atoms):
            type_lst.append(an2tm[self.structures_lst[0].species[tmp_site_idx].Z])
        type_str = "\n".join([str(tmp_type) for tmp_type in type_lst])
        type_str += "\n"
        with open(f"{dir_path}/type.raw", 'w') as f:
            f.write(type_str)
                
        ### Step 1.3. 获取各个 frame 的信息
        box_lst = []    # (num_frames, 9)
        coord_lst = []  # (num_frames, 3*num_atoms)
        force_lst = []  # (num_frames, 3*num_atoms)
        atomic_energy_lst = []  # (num_frames, num_atoms)
        
        for tmp_structure in self.structures_lst:
            box_lst.append(tmp_structure.lattice.matrix.flatten())
            coord_lst.append(tmp_structure.cart_coords)
            force_lst.append([tmp_site.atomic_force for tmp_site in tmp_structure.sites])
            atomic_energy_lst.append([tmp_site.atomic_energy[0] for tmp_site in tmp_structure.sites])
        
        box_array = np.array(box_lst).reshape(self.num_structures, -1)
        coord_array = np.array(coord_lst).reshape(self.num_structures, -1)
        force_array = np.array(force_lst).reshape(self.num_structures, -1)
        tot_energy_array = self.total_energys_array.reshape(self.num_structures, -1)
        virial_array = self.virial_tensors_array.reshape(self.num_structures, -1)
        atomic_energy_array = np.array(atomic_energy_lst)
        
        num_sets = self.num_structures // set_size
        if (set_size * num_sets) < self.num_structures: # 处理剩余的frames
            num_sets += 1
            
        ### Step 1.4. 存储为 .npy 格式
        for tmp_idx in range(num_sets):
            set_stt_idx = tmp_idx * set_size
            set_end_idx = (tmp_idx + 1) * set_size
            set_folder_name = "set.%03d" % tmp_idx  # set.000
            os.makedirs(f"{dir_path}/{set_folder_name}")
            
            ### Note: set_end_idx 超出索引也不会报错
            np.save(file=f"{dir_path}/{set_folder_name}/cell.npy", arr=box_array[set_stt_idx:set_end_idx])  
            np.save(file=f"{dir_path}/{set_folder_name}/atom_coord.npy", arr=coord_array[set_stt_idx:set_end_idx])
            np.save(file=f"{dir_path}/{set_folder_name}/tot_energy.npy", arr=tot_energy_array[set_stt_idx:set_end_idx])
            np.save(file=f"{dir_path}/{set_folder_name}/atom_force.npy", arr=force_array[set_stt_idx:set_end_idx])
            if np.any(self.virial_tensors_array != 0):
                np.save(file=f"{dir_path}/{set_folder_name}/virial.npy", arr=virial_array[set_stt_idx:set_end_idx])
            if np.any(atomic_energy_array != 0):
                np.save(file=f"{dir_path}/{set_folder_name}/atom_energy.npy", arr=atomic_energy_array[set_stt_idx:set_end_idx])
         

    @staticmethod
    def from_indices(
                struct_collection,
                indices_lst:List[int]
                ):
        '''
        Description
        -----------
            1. 根据索引(`indices_lst`)从 deepmd_data_system 中抽取结构作为 sub_deepmd_data_system
        
        Parameters
        ----------
            1. deepmd_data_system: DeepmdDataSystem
                - 
            2. indices_lst: List[int]
                - 索引
        '''
        if max(indices_lst) > len(struct_collection):
            raise IndexError("index in indices_lst is larger than len(DeepmdDataSystem)!!!")
        
        structures_lst = [struct_collection.structures_lst[tmp_index] for tmp_index in indices_lst]
        total_energys_array = np.array([struct_collection.total_energys_array[tmp_index] for tmp_index in indices_lst])
        potential_energys_array = np.array([struct_collection.potential_energys_array[tmp_index] for tmp_index in indices_lst])
        kinetic_energys_array = np.array([struct_collection.kinetic_energys_array[tmp_index] for tmp_index in indices_lst])
        try:    # 不包含 Virial 信息
            virial_tensors_array = np.array([struct_collection.virial_tensors_array[tmp_index] for tmp_index in indices_lst])
        except IndexError as e:
            virial_tensors_array = np.zeros(10);
            
                
        return StructCollection(
                    structures_lst=structures_lst,
                    total_energys_array=total_energys_array,
                    potential_energys_array=potential_energys_array,
                    kinetic_energys_array=kinetic_energys_array,
                    virial_tensors_array=virial_tensors_array)


    def save_all_info(
            self,
            dir_path:str,
            scaling_matrix:List[int]=[3,3,3]):
        '''
        Description
        -----------
            1. 将初始化的信息存入对应image文件夹下 (以npy形式)。
            2. 存储近邻原子信息 -- `nbr_info.npy`
            
        Parameters
        ----------
            1. dir_path: str
                - 输出的文件夹路径
                - 各个frame的存储路径为 `<dir_path>/IMAGE_000`
        
        Note
        ----
            1. 兼容 PWMAT-MLFF 原本的储存格式，每个image/frame存在一个文件夹下
        '''
        ### Step 0. 如果 `dir_path文件夹` 不存在的话，则建立该文件夹
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Folder created: {dir_path}")
        else:
            print(f"Folder already exists: {dir_path}")
        
        ### Step 1. 存储:
        #               1. box.npy
        #               2. coord.npy
        #               3. total_energy.npy
        #               4. kinetic_energy.npy
        #               5. potential_energy.npy
        #               6. atomic_energy.npy
        #               7. atomic_force.npy
        #               8. virial.npy
        num_bits = len(str(self.num_structures))
        for tmp_idx in range(self.num_structures):
            tmp_image_dir_path = os.path.join(dir_path, f"%0{num_bits}d" % tmp_idx)
            ### Step 1.1. 存在-->删除
            if os.path.exists(tmp_image_dir_path):
                shutil.rmtree(tmp_image_dir_path)
            # 创建
            os.mkdir(tmp_image_dir_path)
            
            ### Step 1.2. 保存相关信息
            self.structures_lst[tmp_idx].to(
                                    output_file_format="pwmat",
                                    output_file_path=os.path.join(tmp_image_dir_path, "atom.config"))
            # 1. box.npy
            np.save(
                    file=os.path.join(tmp_image_dir_path, "box.npy"),
                    arr=self.structures_lst[tmp_idx].lattice.matrix
            )
            # 2. coord.npy
            np.save(
                    file=os.path.join(tmp_image_dir_path, "coord.npy"),
                    arr=self.structures_lst[tmp_idx].cart_coords
            )
            # 3\4\5. energy.npy
            np.save(
                    file=os.path.join(tmp_image_dir_path, "energy.npy"), 
                    arr=np.array([
                            self.total_energys_array[tmp_idx],
                            self.potential_energys_array[tmp_idx],
                            self.kinetic_energys_array[tmp_idx]
                    ])
            )
            # 6. atomic_energy
            try:
                np.save(
                        file=os.path.join(tmp_image_dir_path, "atomic_energy.npy"),
                        arr=self.structures_lst[tmp_idx].get_atomic_energy()
                )
            except AttributeError:
                pass
            # 7. atomic_force
            np.save(
                    file=os.path.join(tmp_image_dir_path, "atomic_force.npy"),
                    arr=self.structures_lst[tmp_idx].get_atomic_force()
            )
            # 8. virial
            np.save(
                    file=os.path.join(tmp_image_dir_path, "virial.npy"),
                    arr=self.virial_tensors_array[tmp_idx]
            )
            # 9. atomic_number
            np.save(
                    file=os.path.join(tmp_image_dir_path, "atomic_number.npy"),
                    arr=np.array(self.atomic_numbers_lst)
            )
            # 10. num_atoms
            #num_atoms_dict = dict.fromkeys(self.atomic_numbers_lst, 0)
            #for tmp_atomic_number in self.atomic_numbers_lst:
            #    num_atoms_dict[tmp_atomic_number] += 1
            #np.save(
            #        file=os.path.join(tmp_image_dir_path, "num_atoms.npy"),
            #        arr=np.array(list(num_atoms_dict.values()))
            #)
        
        # 11. nbr_info.npy: 存储 nbr_info 的部分多进程并行
        parameters_lst = [(
                        os.path.join(dir_path, f"%0{num_bits}d" % tmp_idx),
                        self.structures_lst[tmp_idx],
                        scaling_matrix) for tmp_idx in range(self.num_structures)]
        
        with mp.Pool(os.cpu_count()-2) as pool:
            pool.starmap(ParallelFunction.save_struct_nbr, parameters_lst)


    def get_max_num_nbrs_real(
                self,
                rcut:float,
                scaling_matrix:List[int]=[3, 3, 3]):
        '''
        Description
        -----------
            1. 得到该 MOVEMENT 中所有 atom.config 中原子的`最大近邻原子数目`
        '''
        ### Step 1. 得到每个 structure 的 max_nbrs_num_real
        parameters_lst = [(
                    tmp_structure,
                    rcut,
                    scaling_matrix) for tmp_structure in self.structures_lst]
        with mp.Pool(os.cpu_count()-2) as pool:
            max_nbrs_num_real_lst = pool.starmap(
                                        ParallelFunction.get_max_num_nbrs_real,
                                        parameters_lst)
        
        return np.max(max_nbrs_num_real_lst)
    
    
    def get_max_num_nbrs_real_element(
                self,
                rcut:float,
                nbr_elements:List[str],
                scaling_matrix:List[int]):
        '''
        Description
        -----------
            1.
        
        Return
        ------
            1. return_dict
                - e.g. {'Li': 54, 'Si': 29}
                -   近邻的 Li 原子最多有 54 个
                -   近邻的 Si 原子最多有 29 个
        '''
        parameters_lst = [(
                            tmp_structure,
                            rcut,
                            nbr_elements,
                            scaling_matrix) for tmp_structure in self.structures_lst]
        
        ### Step 1. 初始化
        with mp.Pool(os.cpu_count()-2) as pool:
            max_nbrs_num_real_element:Dict[str, int] = pool.starmap(
                                    ParallelFunction.get_max_num_nbrs_real_element,
                                    parameters_lst)
        
        ### Step 2.
        return_dict:Dict[str, int] = dict.fromkeys(
                                        list(max_nbrs_num_real_element[0].keys()),
                                        0
                                    )
        for tmp_dict in max_nbrs_num_real_element:
            for tmp_key in tmp_dict.keys():
                if tmp_dict[tmp_key] > return_dict[tmp_key]:
                    return_dict[tmp_key] = tmp_dict[tmp_key]
               
        return return_dict
        
        
            

class ParallelFunction(object):
    '''
    Description
    -----------
        1. 一些需要多进程并行的函数
    '''    
    @staticmethod
    def save_struct_nbr(
                    tmp_image_dir_path:int,
                    structure:DStructure,
                    scaling_matrix:List[int]):
        '''
        Description
        -----------
            1. 存储 `struct_nbr` 的信息，在 `DeepmdDataSystem.save()` 中调用

        Parameters
        ----------
            1. tmp_image_dir_path: str
                - Image文件夹的路径
            2. structure: DStructure
                - 结构
            3. scaling_matrix: List[int]
                - 扩包倍数
        '''
        # 10. nbr_info.npy
        struct_nbr = StructureNeighborsDescriptor.create(
                    'v1',
                    structure=structure,
                    scaling_matrix=scaling_matrix,
                    reformat_mark=True,
                    coords_are_cartesian=True)

        np.save(
                file=os.path.join(tmp_image_dir_path, "nbrs_atomic_numbers.npy"),
                arr=struct_nbr.key_nbr_atomic_numbers
        )
        np.save(
                file=os.path.join(tmp_image_dir_path, "nbrs_distances.npy"),
                arr=struct_nbr.key_nbr_distances
        )
        np.save(
                file=os.path.join(tmp_image_dir_path, "nbrs_coords.npy"),
                arr=struct_nbr.key_nbr_coords
        )
        
    
    @staticmethod
    def get_max_num_nbrs_real(
                structure:DStructure,
                rcut:float,
                scaling_matrix:List[int]):
        '''
        Description
        -----------
            1. 得到单个结构的 `max_nbrs_num_real`
        '''
        max_num_nbrs_real = StructureNeighborsUtils.get_max_num_nbrs_real(
                                structure=structure,
                                rcut=rcut,
                                scaling_matrix=scaling_matrix,
                                coords_are_cartesian=True)
        
        return max_num_nbrs_real
    
    
    @staticmethod
    def get_max_num_nbrs_real_element(
                structure:DStructure,
                rcut:float,
                nbr_elements:List[str],
                scaling_matrix:List[int]):
        max_nbrs_num_real_element = StructureNeighborsUtils.get_max_num_nbrs_real_element(
                                        structure=structure,
                                        rcut=rcut,
                                        nbr_elements=nbr_elements,
                                        scaling_matrix=scaling_matrix,
                                        coords_are_cartesian=True)
        
        return max_nbrs_num_real_element