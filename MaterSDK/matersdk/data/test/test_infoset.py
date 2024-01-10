import unittest
# python3 -m matersdk.data.test.test_infoset
from ..infoset import InfoSet


class InfoSetTest(unittest.TestCase):
    def test_all(self):
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo1/PWdata/data1/MOVEMENT"
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/SiC/MD/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        
        infosets_atomic_numbers = [3, 14, 7]
        #info_set = InfoSet(file_path=movement_path, file_format="pwmat/movement", infosets_atomic_numbers=infosets_atomic_numbers)
        info_set = InfoSet.from_file(file_path=movement_path, file_format="pwmat/movement")
        
        ### Step 1.
        print("1. ")
        print(info_set.num_frames)
        
        ### Step 2.
        print("2. ")
        info_set.to_dir(dir_path="../test")
        


if __name__ == "__main__":
    unittest.main()