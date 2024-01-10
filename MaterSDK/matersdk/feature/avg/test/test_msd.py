import unittest
import numpy as np

# python3 -m matersdk.feature.avg.test.test_msd
from ..msd import (
                DiffractionIntensity,
                Msd,
                MsdParallelFunction
)
from ....io.pwmat.output.movement import Movement


class DiffractionIntensityTest(unittest.TestCase):
    def test_all(self):
        q = 1
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        #movement_path = "/data/home/liuhanyu/hyliu/pwmat_demo/xhm/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        di_object = DiffractionIntensity(trajectory=movement, q=q)
        
        ### Step 1. calc_di()
        di_array = di_object.calc_di()
        print(di_array)


class MsdTest(unittest.TestCase):
    def all(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        #movement_path = "/data/home/liuhanyu/hyliu/pwmat_demo/xhm/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        
        ### Step 1. calc_msd()
        print()
        print("Step 1. Without subtracting centroid, MSD:")
        msd_object = Msd(trajectory=movement)
        msd_values_lst = msd_object.calc_msd()
        print(np.sum(msd_values_lst))

        ### Step 2. calc_msd_sub_centroid()
        print()
        print("Step 2. After subtracting centroid, MSD:")
        msd_values_lst = msd_object.calc_msd_sub_centroid()
        print(np.sum(msd_values_lst))


class ParallelFunctionTest(unittest.TestCase):
    def all(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure_1 = movement.get_frame_structure(idx_frame=0)
        structure_2 = movement.get_frame_structure(idx_frame=100)
        
        
        ### Step 1. calc_msd_s
        msd_value = MsdParallelFunction.calc_msd_s(
                            structure_1=structure_1,
                            structure_2=structure_2)
        print("Step 1. Without subtracting centroid, MSD for structure_{0} and structure_{1} = {2}".format(0, 100, msd_value))
        
        
        ### Step 2. calc_msd_sub_centroid_s
        msd_value = MsdParallelFunction.calc_msd_sub_centroid_s(
                            structure_1=structure_1,
                            structure_2=structure_2)
        print("Step 1. After subtracting centroid, MSD for structure_{0} and structure_{1} = {2}".format(0, 100, msd_value))



if __name__ == "__main__":
    unittest.main()