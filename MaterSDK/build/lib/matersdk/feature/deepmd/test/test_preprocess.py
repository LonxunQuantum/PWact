import unittest
import numpy as np
from matersdk.io.publicLayer.structure import DStructure
from matersdk.io.pwmat.output.movement import Movement
from matersdk.data.deepmd.data_system import DpLabeledSystem
from matersdk.io.publicLayer.neigh import StructureNeighborsDescriptor
from matersdk.feature.deepmd.se_pair import DpseTildeRPairDescriptor

# python3 -m matersdk.feature.deepmd.test.test_preprocess
from ..preprocess import (
                TildeRNormalizer,
                TildeRPairNormalizer,
                NormalizerPremise,
)


class TildeRNormalizerTest(unittest.TestCase):
    def test_all(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)    
        
        ### Step 1. 
        print()
        print("Step 1. Get DpLabeledSystem:")
        dpsys = DpLabeledSystem.from_trajectory_s(trajectory_object=movement)
        print(dpsys)

        ### Step 2. 
        structure_indices = [*range(10)]
        rcut = 6.5
        rcut_smooth = 6.0
        center_atomic_numbers = [3, 14]
        nbr_atomic_numbers = [3, 14]
        max_num_nbrs = [100, 80]
        scaling_matrix = [3, 3, 3]
        print()
        print("Step 2. Calculate stats (davgs, dstds):")
        tilde_r_normalizer = TildeRNormalizer.from_dp_labeled_system(
                        dp_labeled_system=dpsys,
                        structure_indices=structure_indices,
                        rcut=rcut,
                        rcut_smooth=rcut_smooth,
                        center_atomic_numbers=center_atomic_numbers,
                        nbr_atomic_numbers=nbr_atomic_numbers,
                        max_num_nbrs=max_num_nbrs,
                        scaling_matrix=scaling_matrix)
        davgs, dstds = tilde_r_normalizer.davgs, tilde_r_normalizer.dstds
        print("\nStep 2.1. davgs:", end="\n")
        print(center_atomic_numbers)
        print(davgs)
        print("\nStep 2.2. dstds:", end="\n")
        print(center_atomic_numbers)
        print(dstds)
        
        
        ### Step 3.
        print()
        print("Step 3. ._normalize: ")
        tildeR_dict, tildeR_derivative_dict = tilde_r_normalizer._normalize(structure=movement.get_frame_structure(idx_frame=100))
        print("Step 3.1. The Rij:")
        for tmp_key, tmp_value in tildeR_dict.items():
            print("\t", tmp_key, ": ", tmp_value.shape)
        print("Step 3.2. The derivative of Rij with respect to x, y, z:")
        for tmp_key, tmp_value in tildeR_derivative_dict.items():
            print("\t", tmp_key, ": ", tmp_value.shape)
        
        
        ### Step 4.
        print()
        print("Step 4. normalize:")
        tildeR, tildeR_derivative = tilde_r_normalizer.normalize(structure=movement.get_frame_structure(idx_frame=100))
        print(tildeR.shape, tildeR_derivative.shape)
            
        ### Step 5.
        print()
        print("Step 5. Save TildeRNormalizer to hdf5 file...")
        hdf5_file_path = "/data/home/liuhanyu/hyliu/code/matersdk/test_data/deepmd/normalizer/LiSi_norm.h5"
        tilde_r_normalizer.to(hdf5_file_path=hdf5_file_path)
        
        
        ### Step 6. 
        print()
        print("Step 6. TildeRNormalizer.from_file()")
        new_tilde_r_normalizer = TildeRNormalizer.from_file(hdf5_file_path=hdf5_file_path)
        print(new_tilde_r_normalizer.davgs)
        
        
        ### Step 7.
        print("Step 7. self.__repr__():")
        print(new_tilde_r_normalizer)


class TildRPairNormalizerTest(unittest.TestCase):
    def all(self):
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi.config"
        structure = DStructure.from_file(
                        file_format="pwmat",
                        file_path=atom_config_path)

        scaling_matrix = [3, 3, 3]
        reformat_mark = True
        coords_are_cartesian = True

        center_atomic_number = 3    # Li
        nbr_atomic_numbers = [3, 14]      # Li, Si
        max_num_nbrs = [100, 80]
        rcut = 6.5
        rcut_smooth = 6.0
        
        ### Step 0. 计算某个结构的 TildeR
        ### Step 0.1. Li-Li's Rij
        struct_nbr = StructureNeighborsDescriptor.create(
                        'v1',
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian)
        dpse_tildeR_pair_Li = DpseTildeRPairDescriptor.create(
                        'v1',
                        structure_neighbors=struct_nbr,
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_number=nbr_atomic_numbers[0],
                        rcut=rcut,
                        rcut_smooth=rcut_smooth)
        tildeRs_array_Li = dpse_tildeR_pair_Li.get_tildeR(max_num_nbrs=max_num_nbrs[0])
        
        ### Step 0.2. Li-Si's Rij
        dpse_tildeR_pair_Si = DpseTildeRPairDescriptor.create(
                        'v1',
                        structure_neighbors=struct_nbr,
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_number=nbr_atomic_numbers[1],
                        rcut=rcut,
                        rcut_smooth=rcut_smooth)
        tildeRs_array_Si = dpse_tildeR_pair_Si.get_tildeR(max_num_nbrs=max_num_nbrs[1])
        
        ### Step 0.3. 合并 Li-Li's Rij and Li-Si's Rij
        # (48, 100, 4) + (48, 80, 4) = (48, 180, 4)
        tildeRs_array = np.concatenate([tildeRs_array_Li, tildeRs_array_Si], axis=1)

    
        ### Step 1. 
        print()
        print("Step 1. ")
        normalizer = TildeRPairNormalizer(tildeRs_array=tildeRs_array)
        print("Step 1.1. The davg of environment matrix is :", end='\t')
        print(normalizer.davg)
        print("Step 1.2. The dstd of environment matrix is :", end='\t')
        print(normalizer.dstd)
        
        ### Step 2. 
        new_atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi_32.config"
        new_structure = DStructure.from_file(
                        file_format="pwmat",
                        file_path=new_atom_config_path)
        new_struct_nbr = StructureNeighborsDescriptor.create(
                        'v1',
                        structure=new_structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian)
        new_dpse_tildeR_pair = DpseTildeRPairDescriptor.create(
                        'v1',
                        structure_neighbors=new_struct_nbr,
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_number=nbr_atomic_numbers[1],
                        rcut=rcut,
                        rcut_smooth=rcut_smooth)
        
        ### 计算 Li-Si 的Rij -- max_num_nbrs=80
        # shape = (48, 80, 4)
        new_tildeRs_array = new_dpse_tildeR_pair.get_tildeR(max_num_nbrs=max_num_nbrs[1])
        # shape = (48, 80, 4, 3)
        new_tildeR_derivatives_array = new_dpse_tildeR_pair.calc_derivative(max_num_nbrs=max_num_nbrs[1])
        
        print()
        print("Step 2. Normalize the Rij of DeepPot-SE:")
        print("Step 2.1. Normalized tildeR.shape = ", end="\t")
        print(normalizer.normalize(tildeRs_array=new_tildeRs_array).shape)
        print("Step 2.2. The max value of Rij is : ", end="\t")
        print(np.max(normalizer.normalize(tildeRs_array=new_tildeRs_array)))
        print("Step 2.3. The min value of Rij is : ", end="\t")
        print(np.min(normalizer.normalize(tildeRs_array=new_tildeRs_array)))
        
        print()
        print("Step 3. Normalize the derivative of Rij with respect to x, y, z:")
        normalized_deriv = normalizer.normalize_derivative(tildeR_derivatives_array=new_tildeR_derivatives_array)
        print("Step 3.1. Normalized drivative of tildeR with respect to x, y, z, shape = ", end='\t')
        print(normalized_deriv.shape)

                
class NormalizerPremiseTest(unittest.TestCase):
    def all(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)    
        
        ### Step 1. 
        print()
        print("Step 1. Get DpLabeledSystem:")
        dpsys = DpLabeledSystem.from_trajectory_s(trajectory_object=movement)
        print(dpsys)
        
        ### Step 2. 
        structure_indices = [*range(10)]
        rcut = 6.5
        rcut_smooth = 6.0
        center_atomic_number = 3
        nbr_atomic_numbers = [3, 14]
        max_num_nbrs = [100, 80]
        scaling_matrix = [3, 3, 3]
        print()
        print("Step 2. NormalizerPremise.concat_tildeRs4calc_stats():")
        tildeRs_array = NormalizerPremise.concat_tildeRs4calc_stats(
                            dp_labeled_system=dpsys,
                            structure_indices=structure_indices,
                            rcut=rcut,
                            rcut_smooth=rcut_smooth,
                            center_atomic_number=center_atomic_number,
                            nbr_atomic_numbers=nbr_atomic_numbers,
                            max_num_nbrs=max_num_nbrs,
                            scaling_matrix=scaling_matrix
        )
        print("\nStep 2.1. The shape of tildeRs of {0} structures = {1}".format(
                        10, tildeRs_array.shape
        ))

        ### Step 3.
        print()
        print("Step 3. NormalizerPremise.concat_tildeRs4calc_stats():")
        tildeRs_array = NormalizerPremise.concat_tildeRs4normalize(
                        dp_labeled_system=dpsys,
                        structure_indices=structure_indices,
                        rcut=rcut,
                        rcut_smooth=rcut_smooth,
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_numbers=nbr_atomic_numbers,
                        max_num_nbrs=max_num_nbrs,
                        scaling_matrix=scaling_matrix
        )
        print("\nStep 3.1. The shape of tildeRs of {0} structures = {1}".format(
                        10, tildeRs_array.shape
        ))



if __name__ == "__main__":
    unittest.main()