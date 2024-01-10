import numpy as np
from matersdk.io.pwmat.output.movement import Movement
from matersdk.data.struct_collection import DpLabeledSystem

from matersdk.feature.deepmd.preprocess import TildeRNormalizer


### Part I. Custom parameters
movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
# PWmat-MLFF 取前10帧结构计算`davg`和`dstd`
structure_indices = [*range(10)]    
rcut = 6.5
rcut_smooth = 6.0
# 体系中所有元素的原子序数，最好从小到大排序
center_atomic_numbers = [3, 14]
# 体系中所有元素的原子序数，最好从小到大排序
nbr_atomic_numbers = [3, 14]
# 说明：100代表近邻最多有100个Li；80代表近邻最多有80个Si。需要与`nbr_atomic_numbers`对应
max_num_nbrs = [100, 80]
# bulk: [3,3,3]; slab: [3,3,1]
scaling_matrix = [3, 3, 3]


### Part II. Calc stats
movement = Movement(movement_path=movement_path)
dpsys = DpLabeledSystem.from_trajectory_s(trajectory_object=movement)


tilde_r_normalizer = TildeRNormalizer.from_dp_labeled_system(
                dp_labeled_system=dpsys,
                structure_indices=structure_indices,
                rcut=rcut,
                rcut_smooth=rcut_smooth,
                center_atomic_numbers=center_atomic_numbers,
                nbr_atomic_numbers=nbr_atomic_numbers,
                max_num_nbrs=max_num_nbrs,
                scaling_matrix=scaling_matrix
)

davgs, dstds = tilde_r_normalizer.davgs, tilde_r_normalizer.dstds
print("\nStep 1. davgs = ")
print(davgs)
print("\nStep 2. dstds = ")
print(dstds)