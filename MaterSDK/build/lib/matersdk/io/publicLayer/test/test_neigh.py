import unittest

# python3 -m matersdk.io.publicLayer.test.test_neigh
from ..structure import DStructure
from ..neigh import StructureNeighborsDescriptor
from ..neigh import StructureNeighborsUtils
from ...pwmat.output.movement import Movement


class StructureNeighborsV1Test(unittest.TestCase):
    def all(self):
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        #atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi.config"
        scaling_matrix = [5, 5, 1]
        #scaling_matrix = [3, 3, 3]
        rcut = 3.2
        #rcut = 6.5
        reformat_mark = True
        coords_are_cartesian = True   
        
        structure = DStructure.from_file(
                        file_format="pwmat",
                        file_path=atom_config_path)
        neighbors_v1 = StructureNeighborsDescriptor.create(
                        "v1",
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian,
                        )
        
        print()
        print("Step 1. primitive_cell 中原子的近邻原子情况:")
        print("\t1.1. The number of atoms in primitive cell:\t", len(neighbors_v1.structure.species))
        print("\t1.2. The shape of key_nbr_species:\t", neighbors_v1.key_nbr_atomic_numbers.shape)
        print("\t1.3. The shape of key_nbr_distances:\t", neighbors_v1.key_nbr_distances.shape)
        print("\t1.4. The shape of key_nbr_coords:\t", neighbors_v1.key_nbr_coords.shape)
        #print(neighbors_v1.key_nbr_atomic_numbers)



class StructureNeighborUtilsTest(unittest.TestCase):
    def test_all(self):
        '''
        ### Step 0.1. 2D
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        #atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi.config"
        scaling_matrix = [5, 5, 1]
        #scaling_matrix = [3, 3, 3]
        rcut = 3.2
        #rcut = 6.5
        coords_are_cartesian = True   
        
        structure = DStructure.from_file(
                        file_format="pwmat",
                        file_path=atom_config_path)
        '''
        
        ### Step 0.1. Li2Si
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        scaling_matrix = [3, 3, 3]
        rcut = 5
        coords_are_cartesian = True   
        reformat_mark = True
        # 343st corresponds to image_005
        structure = movement.get_frame_structure(idx_frame=343)
    
        
        ### Step 1.
        max_num_nbrs_real = StructureNeighborsUtils.get_max_num_nbrs_real(
                                    structure=structure,
                                    rcut=rcut,
                                    scaling_matrix=scaling_matrix,
                                    coords_are_cartesian=coords_are_cartesian)
        print("Step 1. 此 atom.config 在截断半径 {0} 内，最大近邻原子数（不包括中心原子自身）为:".format(rcut), end="\t")
        print(max_num_nbrs_real)
        
        print("Step 2. 此 atom.config 在截断半径 {0} 内，最大近邻各元素的元素数（不包括中心原子自身）为:".format(rcut), end="\t")
        max_num_nbrs_real = StructureNeighborsUtils.get_max_num_nbrs_real_element(
                                    structure=structure,
                                    rcut=rcut,
                                    nbr_elements=["Li", "Si"],
                                    scaling_matrix=scaling_matrix,
                                    coords_are_cartesian=coords_are_cartesian)
        print(max_num_nbrs_real)
        
        print("Step 3. 获取 neigh_list (这个函数为了PWmatMLFF的inference制定):")
        struct_nbr = StructureNeighborsDescriptor.create(
                        "v1",
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian,
                        )
        center_atomic_number = 3
        nbr_atomic_number = 15
        ### Step 3.1. 
        print("Step 3.1. With max_num_nbrs = 100:", end='\t')
        dR_neigh_list = StructureNeighborsUtils._get_nbrs_indices(
                            struct_nbr=struct_nbr,
                            center_atomic_number=center_atomic_number,
                            nbr_atomic_number=nbr_atomic_number,
                            max_num_nbrs=100)
        print(dR_neigh_list.shape)
        print("Step 3.2. Without max_num_nbrs:", end='\t')
        dR_neigh_list = StructureNeighborsUtils._get_nbrs_indices(
                            struct_nbr=struct_nbr,
                            center_atomic_number=center_atomic_number,
                            nbr_atomic_number=nbr_atomic_number)
        print(dR_neigh_list.shape)
        
        
        ### Step 4. 
        print()
        print("Step 4. After concat, 获取 neigh_list:")
        dR_neigh_list = StructureNeighborsUtils.get_nbrs_indices(
                        struct_nbr=struct_nbr,
                        center_atomic_numbers=[3, 14],
                        nbr_atomic_numbers=[3, 14],
                        max_num_nbrs=[100, 80]
        )
        print(dR_neigh_list.shape)

    
if __name__ == "__main__":
    unittest.main()