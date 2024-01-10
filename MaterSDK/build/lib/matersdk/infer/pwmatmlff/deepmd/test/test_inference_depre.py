import unittest
import numpy as np

# python3 -m matersdk.infer.pwmatmlff.deepmd.test.test_inference
from .....io.publicLayer.structure import DStructure
from ..inference import FFInfer


class FFInferTest(unittest.TestCase):
    def test_all(self):
        atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/feature/movement/LiSi.config"
        structure = DStructure.from_file(
                            file_path=atom_config_path,
                            file_format="pwmat")
        hdf5_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/infer/pwmatmlff/deepmd/test.h5"
        rcut = 6.5
        rcut_smooth = 6.0
        scaling_matrix = [3, 3, 3]
        max_num_nbrs_dict = {3: 100, 14: 80}
        davgs_dict = {
            3: np.ndarray([0.08375034, 0., 0., 0.]),
            14: None
        }
        dstds_dict = {
            3: np.ndarray([0.11677445, 0.08296664, 0.08296664, 0.08296664]),
            14: None
        }
        
        ff_infer = FFInfer(
                    hdf5_path=hdf5_path,
                    rcut=rcut,
                    rcut_smooth=rcut_smooth,
                    max_num_nbrs_dict=max_num_nbrs_dict,
                    davgs_dict=davgs_dict,
                    dstds_dict=dstds_dict)

        ### Step 1.
        print()
        print("Step 1. The atomic numbers in system:", end='\t')
        print(ff_infer._get_atomic_numbers(structure=structure))
        #ff_infer.featurize(structure=structure)
        
        ### Step 2. 
        print()
        print("Step 2. model_params_dict:")
        for tmp_key, tmp_value in ff_infer.model_params_dict.items():
            if type(tmp_value) == np.ndarray:
                print(tmp_key, " .shape= ", tmp_value.shape)
            else:
                print(tmp_key, " = ", tmp_value)
        
        ### Step 3. 
        print()
        print("Step 3. The shape of tildeR:")
        tildeR_dict = ff_infer.calc_tildeR(structure=structure, scaling_matrix=scaling_matrix)
        for tmp_key, tmp_value in tildeR_dict.items():
            print(tmp_key, " .shape= ", tmp_value.shape)

if __name__ == "__main__":
    unittest.main()