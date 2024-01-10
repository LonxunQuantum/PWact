import unittest
import numpy as np

# python3 -m matersdk.io.pwmat.utils.test.test_acstrextractor
from ...output.movement import Movement
from ..acextractor import ACstrExtractor


class ACstrExtractorTest(unittest.TestCase):
    def test_all(self):
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo1/PWdata/data1/MOVEMENT"
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/SiC/MD/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        idx_frame = 2 # 帧数从 0 开始计数
        
        ### Step 0. 得到某 frame 的 string
        atom_config_string = movement._get_frame_str(idx_frame=idx_frame)
        #print(atom_config_string)

        ### Step 1. 得到体系内的原子数目
        atom_config_str_extractor = ACstrExtractor(
                                        atom_config_str=atom_config_string)
        print()
        print("Step 1. The number of atom in system:", end="\t")
        print(atom_config_str_extractor.get_num_atoms())
        
        ### Step 2. 得到体系的 basis vectors
        print()
        print("Step 2. The basis vectors of system:") 
        print(atom_config_str_extractor.get_basis_vectors())
           
        
        ### Step 3. 得到体系的原子序数 (重复的)
        print()
        print("Step 3. The types in system:")
        print(atom_config_str_extractor.get_types())
        
        
        ### Step 4. 得到体系的坐标 (np.array 形式)
        print()
        print("Step 4. The frac coords of atoms in system:")
        print(atom_config_str_extractor.get_coords())
        
        
        ### Stpe 5. 得到体中各个原子的磁矩
        print()
        print("Step 5. The magnetic moment of atoms in system:")
        #print(atom_config_str_extractor.get_magmoms())


        ### Stpe 6. 得到体中各个原子的能量
        print()
        print("Step 6. The atomic energy of atoms in system:")
        print(atom_config_str_extractor.get_eatoms())        
        
        ### Stpe 7. 得到体系总能
        print()
        print("Step 7. The total energy of atoms in system:")
        print(atom_config_str_extractor.get_etot())

        ### Step 8. 得到体系中各个原子的受力
        print()
        print("Step 8. The atomic forces of atoms in system:")
        print(atom_config_str_extractor.get_fatoms())
    
    
        ### Step 9. 得到体系的virial
        print()
        print("Step 9. The atomic forces of atoms in system:")
        print(atom_config_str_extractor.get_virial())     

        


if __name__ == "__main__":
    unittest.main()