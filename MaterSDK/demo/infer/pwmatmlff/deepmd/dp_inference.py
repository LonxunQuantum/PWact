from matersdk.io.pwmat.output.movement import Movement
from matersdk.data.struct_collection import DpLabeledSystem
from matersdk.feature.deepmd.preprocess import TildeRNormalizer
from matersdk.infer.pwmatmlff.deepmd.inference import DpInfer

import numpy as np
import torch
print("Torch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("Number of CUDA: ", torch.cuda.device_count())
for idx_cpu in range(torch.cuda.device_count()):
    print("torch.cuda.get_device_name : ", torch.cuda.get_device_name(idx_cpu))



### Step 0. 自定义参数
movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"  # 产生训练集、验证集的 MOVEMENT
pt_file_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/demo2/record/checkpoint.pt"
device = "cuda:0"
rcut = 6.0                      # Rmax
rcut_smooth = 5.5               # Rmin
scaling_matrix = [3, 3, 3]      # 三维体系：[3, 3, 3]; 二维体系: [3, 3, 1]
davg = None                     # 设置为 None 时，可以从 MOVEMENT 自动计算
dstd = None                     # 设置为 None 时，可以从 MOVEMENT 自动计算
center_atomic_numbers = [3, 14] # 体系内所有元素的原子序数，从小到大排列
nbr_atomic_numbers = [3, 14] # 体系内所有元素的原子序数，从小到大排列
max_num_nbrs = [100, 100]    # 近邻原子的最大数目，与 nbr_atomic_numbers 对应
reformat_mark = True            # 永远都是True
coords_are_cartesian = True # 永远都是True
movement = Movement(movement_path=movement_path)
new_structure = movement.get_frame_structure(idx_frame=500) # 需要做 inference 的结构


### Step . 计算 `relative_coords`, 用于检测 `virial tensor`
from matersdk.io.publicLayer.neigh import StructureNeighborsDescriptor
from matersdk.feature.deepmd.premise import DpFeaturePairPremiseDescriptor

struct_nbr = StructureNeighborsDescriptor.create(
                "v1",
                structure=new_structure,
                rcut=rcut,
                scaling_matrix=scaling_matrix,
                reformat_mark=reformat_mark,
                coords_are_cartesian=coords_are_cartesian)
dp_feature_pair_premise = DpFeaturePairPremiseDescriptor.create(
                                "v1",
                                structure_neighbors=struct_nbr)

pair_rcs = []
for tmp_center_an in center_atomic_numbers:
    center_pair_rcs = []
    for tmp_neigh_idx, tmp_neigh_an in enumerate(nbr_atomic_numbers):
        dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc = \
                dp_feature_pair_premise.extract_feature_pair(
                                center_atomic_number=tmp_center_an,
                                nbr_atomic_number=tmp_neigh_an
                )  
                      
        pair_rc = np.zeros(shape=(dp_feature_pair_rc.shape[0], max_num_nbrs[tmp_neigh_idx], 3))
        for ii in range(dp_feature_pair_rc.shape[0]): 
            pair_rc[:, :dp_feature_pair_rc.shape[1], :] = dp_feature_pair_rc
        
        print("center_neigh_rc : ", dp_feature_pair_rc.shape, pair_rc.shape)
        center_pair_rcs.append(pair_rc)
        
    center_pair_rc = np.concatenate(center_pair_rcs, axis=1)
    pair_rcs.append(center_pair_rc)


pair_rcs_array = np.concatenate(pair_rcs, axis=0)   # shape = (num_centers, num_nbrs, 3)
print("relative_coords : ", pair_rcs_array.shape)

np.save(file="demo_infer_result/relative_coords.npy", arr=pair_rcs_array)




### Step 1. 计算 `davg`, `dstd`

### Step 1.1. 计算 Rij 的 davg, dstd
dpsys = DpLabeledSystem.from_trajectory_s(trajectory_object=movement)
tildeR_normalizer = TildeRNormalizer.from_dp_labeled_system(
                    dp_labeled_system=dpsys,
                    structure_indices=[*(range(10))],
                    rcut=rcut,
                    rcut_smooth=rcut_smooth,
                    center_atomic_numbers=center_atomic_numbers,
                    nbr_atomic_numbers=nbr_atomic_numbers,
                    max_num_nbrs=max_num_nbrs,
                    scaling_matrix=scaling_matrix
)
davg, dstd = tildeR_normalizer.davgs, tildeR_normalizer.dstds


### Step 2. 
dp_infer = DpInfer(
    pt_file_path=pt_file_path,
    device=device,
    rcut=rcut,
    rcut_smooth=rcut_smooth,
    davg=davg,
    dstd=dstd,
    center_atomic_numbers=center_atomic_numbers,
    nbr_atomic_numbers=nbr_atomic_numbers,
    max_num_nbrs=max_num_nbrs,
    scaling_matrix=scaling_matrix
)


e_tot, e_atoms, f_atoms, virial = dp_infer.infer(structure=new_structure)
print( "S 2.1. e_tot = {0} eV".format(e_tot.item()) )
print( "S 2.2. e_atoms.shape = ", e_atoms.shape )
print( "S 2.3. f_atoms.shape = ", f_atoms.shape )
print( "S 2.4. virial = \n", virial )


np.save(file="demo_infer_result/e_tot.npy", arr=e_tot)
np.save(file="demo_infer_result/e_atoms.npy", arr=e_atoms)
np.save(file="demo_infer_result/f_atoms.npy", arr=f_atoms)
np.save(file="demo_infer_result/virial.npy", arr=virial)