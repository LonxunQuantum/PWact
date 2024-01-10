import unittest

# python3 -m matersdk.calculation.kpoints.test.test_kmesh
from ..kmesh import KMesh


class KMeshTest(unittest.TestCase):
    def test_get_lattice_info(self):
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/kpath/atom.config"

        kmesh = KMesh(
            file_format=file_format,
            file_path=file_path
            )
        kmesh.get_lattice_info()



    def test_get_kmesh(self):
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/kpath/atom.config"
        #file_format = "pwmat"
        #file_path = "/Users/mac/我的文件/Mycode/new/new2/matersdk/test_data/atom_config/atom.config"
        density = 1
        density = 0.04

        kmesh = KMesh(
            file_format=file_format,
            file_path=file_path
        )
        print("KMesh when density = {0} (unit: 2pi/Angstrom)".format(density))
        print(kmesh.get_kmesh(density))

if __name__ == "__main__":
    unittest.main()