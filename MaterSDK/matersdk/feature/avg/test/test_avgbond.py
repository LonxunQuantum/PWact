import unittest

# python3 -m matersdk.feature.avg.test.test_avgbond
from ..avgbond import AvgBond, PairBond
from ....io.pwmat.output.movement import Movement
from ....io.publicLayer.neigh import StructureNeighborsDescriptor


class AvgBondTest(unittest.TestCase):
    def frames_avg_bond(self):
        #movement_path = "/data/home/liuhanyu/hyliu/data_for_test/bondfft_GeTe/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        #element_1 = "Ge"
        #element_2 = "Te"
        element_1 = "Li"
        element_2 = "Si"
        rcut = 3.2
        
        bondfft = AvgBond(
                        movement_path=movement_path,
                        element_1=element_1,
                        element_2=element_2,
                        rcut=rcut)
        ### Step 1.
        print( len(bondfft.frames_lst) )
        
        ### Step 2. 
        print(bondfft.get_frames_avg_bond())
        
        
    def frame_avg_bond(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure = movement.get_frame_structure(idx_frame=0)
        atomic_number_1 = 3
        atomic_number_2 = 14
        rcut = 3.4
        scaling_matrix = [3, 3, 3]  # 二维材料:[3, 3, 1]; 三维材料:[3, 3, 3]
        
        struct_neigh = StructureNeighborsDescriptor.create(
                        'v1',
                        structure,
                        rcut,
                        scaling_matrix)
        
        avg_bond_length = AvgBond.get_avg_bond_length(
                    struct_neigh=struct_neigh,
                    atomic_number_1=atomic_number_1, 
                    atomic_number_2=atomic_number_2,
        )
        
        print("在 {0} 埃内，{1}-{2} 的平均键长为 {3} 埃".format(rcut, atomic_number_1, atomic_number_2, avg_bond_length))


    def test_frame_bonds_lst(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure = movement.get_frame_structure(idx_frame=0)
        atomic_number_1 = 3
        atomic_number_2 = 3
        atomic_number_3 = 3
        rcut = 3.4
        scaling_matrix = [3, 3, 3]
        angle_standard = 120
        angle_epsilon = 5
        
        struct_neigh = StructureNeighborsDescriptor.create(
                        'v1',
                        structure,
                        rcut,
                        scaling_matrix)

        result_array = AvgBond.get_bond_lengths_lst_according2angle(
                            struct_neigh=struct_neigh,
                            atomic_number_1=atomic_number_1,
                            atomic_number_2=atomic_number_2,
                            atomic_number_3=atomic_number_3,
                            angle_standard=angle_standard,
                            angle_epsilon=angle_epsilon,
        )
        print(result_array)
    



class PairBondTest(unittest.TestCase):
    def all(self):
        #movement_path = "/data/home/liuhanyu/hyliu/data_for_test/bondfft_GeTe/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        atom1_idx = 0
        atom2_idx = 1
        
        pairbond = PairBond(
                            movement_path=movement_path,
                            atom1_idx=atom1_idx,
                            atom2_idx=atom2_idx)
        #print( len(pairbond.frames_lst) )
        
        #print(pairbond.get_frames_pair_bond())
        


if __name__ == "__main__":
    unittest.main()