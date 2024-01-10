import unittest

# python3 -m matersdk.feature.deepmd.test.test_premise
from ....io.publicLayer.structure import DStructure
from ....io.publicLayer.neigh import StructureNeighborsDescriptor
from ..premise import DpFeaturePairPremiseDescriptor



class DpFeatureTest(unittest.TestCase):        
    def test_all_v1(self):
        ### Step 0.1. 
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        #atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi.config"
        scaling_matrix = [5, 5, 1]
        #scaling_matrix = [3, 3, 3]
        reformat_mark = True
        coords_are_cartesian = True
        rcut = 3.2
        center_atomic_number = 42
        nbr_atomic_number = 42
        
        
        structure = DStructure.from_file(
                        file_format="pwmat", 
                        file_path=atom_config_path)
        neighbors = StructureNeighborsDescriptor.create(
                        "v1",
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian)

        ### Step 1. 抽取一对 "中心原子-近邻原子" 的 DpFeaturePairPremise
        print()
        print("Step 1. extract_feature:")
        dp_feature = DpFeaturePairPremiseDescriptor.create(
                        "v1",
                        structure_neighbors=neighbors)
        
        dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc = \
            dp_feature.extract_feature_pair(
                        center_atomic_number=center_atomic_number,
                        nbr_atomic_number=nbr_atomic_number
            )
        
        print("1.1. Atomic number -- dp_feature_pair_an.shape:", end='\t')
        print(dp_feature_pair_an.shape)
        print()
        print("1.2. Distance -- dp_feature_pair_d.shape:", end='\t')
        print(dp_feature_pair_d.shape)
        print()
        print("1.3. Coords -- dp_feature_pair_rc:", end='\t')
        print(dp_feature_pair_rc.shape)
        
        
        ### Step 2. expand_rc: For siyu's PWmatMLFF
        print()
        print("Step 2. ")
        expanded_rc = dp_feature.expand_rc(
                    center_atomic_number=center_atomic_number,
                    nbr_atomic_number=nbr_atomic_number,
                    max_num_nbrs=100
        )
        print(expanded_rc.shape)

        

if __name__ == "__main__":
    unittest.main()