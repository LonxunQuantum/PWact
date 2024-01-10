import unittest


# python3 -m matersdk.infer.pwmatmlff.deepmd.test.test_preprocessor
from .....io.pwmat.output.movement import Movement
from .....data.deepmd.data_system import DpLabeledSystem
from .....feature.deepmd.preprocess import TildeRNormalizer
from ..preprocess import InferPreprocessor


class InferPreprocessorTest(unittest.TestCase):
    def test_all(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"  # 产生训练集、验证集的 MOVEMENT
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
        ### 计算 Rij 的 davg, dstd
        if (not davg) or (not dstd):
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

        infer_preprocessor = InferPreprocessor(
                    structure=movement.get_frame_structure(idx_frame=100),
                    rcut=rcut,
                    rcut_smooth=rcut_smooth,
                    scaling_matrix=scaling_matrix,
                    davg=davg,
                    dstd=dstd,
                    center_atomic_numbers=center_atomic_numbers,
                    nbr_atomic_numbers=nbr_atomic_numbers,
                    max_num_nbrs=max_num_nbrs,
                    reformat_mark=reformat_mark,
                    coords_are_cartesian=coords_are_cartesian
        )
        
        
        ### Step 1. 
        ImageDR = infer_preprocessor.expand_rc()
        print("1. ImageDR.shape = ", ImageDR.shape)
        
        
        ### Step 2.
        Ri, Ri_d = infer_preprocessor.expand_tildeR()
        print("2. Ri.shape = ", Ri.shape)
        print("3. Ri_d.shape =", Ri_d.shape)

        ### Step 3.
        list_neigh = infer_preprocessor.expand_list_neigh()
        print("4. list_neigh.shape = ", list_neigh.shape)
        
        ### Step 4. 
        natoms_img = infer_preprocessor.expand_natoms_img()
        print("5. natoms_img = ", natoms_img.shape)

if __name__ == "__main__":
    unittest.main()