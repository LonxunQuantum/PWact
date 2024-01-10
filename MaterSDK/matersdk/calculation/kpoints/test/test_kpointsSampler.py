import unittest
import numpy as np

# python3 -m matersdk.calculation.kpoints.test.test_kpointsSampler
from ..kpointsSampler import KPointsSampler
from ....io.publicLayer.structure import DStructure



class KpointsSamplerTest(unittest.TestCase):
    def test_get_kpoints(self):
        atom_config_path = "/Users/mac/我的文件/Mycode/new/new2/matersdk/test_data/atom_config/atom.config"
        structure = DStructure.from_file(
                        file_path=atom_config_path,
                        file_format="pwmat",
                        coords_are_cartesian=False,
                        )
        kmesh = np.array([2, 2, 2])
        is_shift = np.array([0, 0, 0])
        symprec = 1e-3
        angle_tolerance = 5.0

        kpoints_sampler = KPointsSampler(
                            structure=structure,
                            kmesh=kmesh,
                            is_shift=is_shift,
                            symprec=symprec,
                            angle_tolerance=angle_tolerance
                        )
        print(kpoints_sampler.get_kpoints())


    def test_get_num_kpoints(self):
        atom_config_path = "/Users/mac/我的文件/Mycode/new/new2/matersdk/test_data/atom_config/atom.config"
        structure = DStructure.from_file(
                        file_path=atom_config_path,
                        file_format="pwmat",
                        coords_are_cartesian=False,
                        )
        kmesh = np.array([2, 2, 2])
        is_shift = np.array([0, 0, 0])
        symprec = 1e-3
        angle_tolerance = 5.0

        kpoints_sampler = KPointsSampler(
                            structure=structure,
                            kmesh=kmesh,
                            is_shift=is_shift,
                            symprec=symprec,
                            angle_tolerance=angle_tolerance
                        )

        print("Numbers of KPoints: ", kpoints_sampler.get_num_kpoints())



if __name__ == "__main__":
    unittest.main()