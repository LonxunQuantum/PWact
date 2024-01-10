import unittest

# python3 -m matersdk.feature.avg.test.test_avgangle
from ..avgangle import AvgAngle
from ....io.pwmat.output.movement import Movement
from ....io.publicLayer.neigh import StructureNeighborsDescriptor


class AvgAngleTest(unittest.TestCase):
    def test_get_avg_bond(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure = movement.get_frame_structure(idx_frame=0)
        atomic_number_1 = 3
        atomic_number_2 = 3
        atomic_number_3 = 3
        rcut = 3.4
        scaling_matrix = [3, 3, 3]
        
        struct_neigh = StructureNeighborsDescriptor.create(
                        'v1',
                        structure,
                        rcut,
                        scaling_matrix)
        
        avg_angle = AvgAngle.get_avg_bond(
                    struct_neigh=struct_neigh,
                    atomic_number_1=atomic_number_1,
                    atomic_number_2=atomic_number_2,
                    atomic_number_3=atomic_number_3)

        print(avg_angle)
        
        
if __name__ == "__main__":
    unittest.main()