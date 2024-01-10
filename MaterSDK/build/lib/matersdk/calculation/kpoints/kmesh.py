# python3 -m matersdk.calculation.kpoints.kmesh
import numpy as np
from ...io.publicLayer.structure import DStructure


class KMesh(object):
    def __init__(self,
                file_format:str,
                file_path:str,
                ):
        self.structure = DStructure.from_file(
                                        file_format=file_format,
                                        file_path=file_path,
                                        coords_are_cartesian=False,
                                        )
    
    def get_lattice_info(self):
        # 1. 输出实空间点阵的基矢
        print("The LATTICE in Angstrom")
        print("{0:-^23}".format("-"))
        print(self.structure.lattice)
        print("")

        # 2. 实空间基矢的长度
        print("The Length in Angstrom")
        print("{0:-^22}".format("-"))
        print(self.structure.lattice.abc)
        print("")

        # 3. 输出倒易空间中的基矢 (unit: 2pi/Angstrom)
        print("The Reciprocal LATTICE in 2pi/Angstrom")
        print("{0:-^38}".format("-"))
        print(self.structure.lattice.reciprocal_lattice)
        print("")

        # 4. 倒易空间基矢的长度
        print("The Reciprocal Length in 2pi/Angstrom")
        print("{0:-^37}".format("-"))
        print(self.structure.lattice.reciprocal_lattice.abc)

    
    def get_kmesh(self, density:float):
        '''
        Description
        -----------
            1. 根据density, 得到 Kpoint 网格 (K-MESH)
        
        Note
        ----
            1. density 的单位: 2pi/Angstrom
        '''
        if density == 0:
            return np.array([1, 1, 1])

        # reciprocal_basis_vectors: 倒易格子的基矢 (unit: 2pi/Angstrom)
        reciprocal_basis_vectors_in_2pi = self.structure.lattice.reciprocal_lattice.matrix

        # reciprocal_basis_lengths: 倒易格子的长度 (unit: 2pi/Angstrom)
        reciprocal_basis_lengths_in_2pi = np.sqrt(
                                    np.sum(np.power(reciprocal_basis_vectors_in_2pi, 2),
                                    axis=1),
                                    )
        # 计算 k-mesh
        kmesh = reciprocal_basis_lengths_in_2pi / (density * 2 * np.pi)
        kmesh = np.round(kmesh)

        ### Note: 如果有真空，真空方向 kmesh 为 1
        for idx_direction in range(3):
            if self.structure.judge_vacuum_exist()[idx_direction]:
                kmesh[idx_direction] = 1
        
        ### Note: 如果某方向kmesh为0，那么将这个方向kmesh设置为1
        for idx_direction in range(3):
            if kmesh[idx_direction] == 0:
                kmesh[idx_direction] = 1

        return kmesh


if __name__ == "__main__":
    kmesh = KMesh(
            file_format="pwmat",
            file_path="/Users/mac/我的文件/Mycode/new/new2/matersdk/test_data/atom_config/atom.config"
            )
    kmesh.get_lattice_info()
    print( kmesh.get_kmesh(0.03) )