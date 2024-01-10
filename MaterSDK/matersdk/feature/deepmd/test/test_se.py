import unittest

# python3 -m matersdk.feature.deepmd.test.test_se
from ....io.publicLayer.structure import DStructure
from ....io.publicLayer.neigh import StructureNeighborsDescriptor
from ..se import DpseTildeRDescriptor


class DpseTildeRTest(unittest.TestCase):
    def test_all(self):
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        scaling_matrix = [5, 5, 1]
        reformat_mark = True
        coords_are_cartesian = True
        
        center_atomic_numbers_lst = [16, 42]
        nbr_atomic_numbers_lst = [16, 42]
        sel = [20, 15]  # 20 for 16(S); 15 for 42(Mo)
        rcut = 3.2
        rcut_smooth = 3.0
        
        
        structure = DStructure.from_file(
                        file_format="pwmat",
                        file_path=atom_config_path)
        neighbors = StructureNeighborsDescriptor.create(
                        'v1',
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian)

        ### Step 1. 
        print("\nStep 1. The shape of tildeR = ", end="\t")
        dpse_tildeR = DpseTildeRDescriptor.create(
                        'v1',
                        structure_neighbors=neighbors,
                        center_atomic_numbers_lst=center_atomic_numbers_lst,
                        nbr_atomic_numbers_lst=nbr_atomic_numbers_lst,
                        sel=sel,
                        rcut=rcut,
                        rcut_smooth=rcut_smooth)
        
        tilde_r_tot = dpse_tildeR.get_tildeR()
        print(tilde_r_tot)
        


if __name__ == "__main__":
    unittest.main()