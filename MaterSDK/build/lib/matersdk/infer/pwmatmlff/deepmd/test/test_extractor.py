import unittest
import os
import numpy as np

# python3 -m matersdk.infer.pwmatmlff.deepmd.test.test_extractor
from ..extractor import FFExtractor


class FFExtractorTest(unittest.TestCase):
    def test_all(self):
        pt_path = "/data/home/liuhanyu/hyliu/pwmat_demo/ff_files/demo2/checkpoint.pth.tar"
        hdf5_path = "./test.h5"
        num_embedding_nets = 3
        num_embedding_layers = 3
        num_fitting_nets = 2
        num_fitting_layers = 3
        
        ### Step 1.
        print()
        print("Step 1. Get keys from model[state_dict]:")
        ff_extractor = FFExtractor(
                        pt_path=pt_path,
                        num_embedding_nets=num_embedding_nets,
                        num_embedding_layers=num_embedding_layers,
                        num_fitting_nets=num_fitting_nets,
                        num_fitting_layers=num_fitting_layers,
        )
        model = ff_extractor.model
        #print("state_dict.fitting_net.1.weights.weight1:")
        #print(model["state_dict"]['fitting_net.1.weights.weight1'].shape)
        #print("state_dict.fitting_net.1.bias.bias1:")
        #print(model["state_dict"]['fitting_net.1.bias.bias1'])
        print(ff_extractor.model_state_dict.keys())
        
        ### Step 2.
        print()
        print("Step 2. Save to HDF5 file...")
        ff_extractor.save_hdf5_file(hdf5_path=hdf5_path)
        
        
        ### Step 3.
        print()
        print("Step 3. Read HDF5 file.")
        import h5py
        hdf5_file = h5py.File(hdf5_path, 'r')
        print("Step 3.1. hdf5_file.keys() = ")
        print(hdf5_file.keys())
        print("Step 3.2. The type of value give hdf5_file.keys():")
        #for tmp_key in hdf5_file.keys():
        #    print(type(hdf5_file[tmp_key][()]))
        
        
        ### Step 4. 
        print()
        print("Step 4. get_hdf5_dict():")
        hdf5_dict = FFExtractor.get_hdf5_dict(hdf5_path=hdf5_path)
        print("Step 4.1. hdf5_dict.keys()")
        for tmp_key, tmp_value in hdf5_dict.items():
            if type(tmp_value) == np.ndarray:
                print("{0} .shape: = {1}".format(tmp_key, tmp_value.shape))
        print("Step 4.2. embedding_sizes = ", end='\t')
        print(hdf5_dict["embedding_sizes"])
        print("Step 4.3. fitting_sizes = ", end='\t')
        print(hdf5_dict["fitting_sizes"])
        
        
        os.remove(path=hdf5_path)
        
        

if __name__ == "__main__":
    unittest.main()