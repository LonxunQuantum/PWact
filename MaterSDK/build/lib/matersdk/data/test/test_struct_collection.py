import unittest

# python3 -m matersdk.data.test.test_struct_collection
from ..struct_collection import StructCollection
from ...io.pwmat.output.movement import Movement


movement_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/PWMLFF/example/SiC/1_300_MOVEMENT"


class StructCollectionTest(unittest.TestCase):
    def test_common(self):
        movement = Movement(movement_path=movement_path)
        struct_collection = StructCollection.from_trajectory_s(trajectory_object=movement)
        save_dir_path = "./test_data"
        
        ### Step 1. 
        print(struct_collection)

        ### Step 2. 
        struct_collection.to(
                        dir_path=save_dir_path, 
                        set_size=30)
        
        ### Step 3.
        train_struct_collection = StructCollection.from_indices(
                        struct_collection=struct_collection, 
                        indices_lst=[*range(10)])
        print(train_struct_collection)

if __name__ == "__main__":
    unittest.main()