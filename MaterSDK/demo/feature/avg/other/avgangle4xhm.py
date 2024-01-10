import numpy as np
from matersdk.io.pwmat.output.movement import Movement
from matersdk.io.publicLayer.neigh import StructureNeighborsDescriptor
from matersdk.feature.avg.avgangle import AvgAngle


movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
movement = Movement(movement_path=movement_path)
idx_frame = 0               # 取第几步的结构
structure = movement.get_frame_structure(idx_frame=idx_frame)
atomic_number_1 = 3         # `中心原子`的原子序数
atomic_number_2 = 3         # 近邻原子_1，最后统计的是`中心原子 - 近邻原子_1` 的键长
atomic_number_3 = 3         # 近邻原子_2
rcut = 3.4                  # 截断半径
scaling_matrix = [3, 3, 3]  # 三维体系: [3, 3, 3]; 二维体系: [3, 3, 1]


struct_neigh = StructureNeighborsDescriptor.create(
                'v1',
                structure,
                rcut,
                scaling_matrix)
avg_angle, result = AvgAngle.get_avg_bond(
            struct_neigh=struct_neigh,
            atomic_number_1=atomic_number_1,
            atomic_number_2=atomic_number_2,
            atomic_number_3=atomic_number_3)

print("Result = ", result)