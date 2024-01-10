import unittest
import warnings

# python3 -m matersdk.calculation.kpath.test.test_kpathSampler
from ....io.publicLayer.structure import DStructure
from ..kpathSampler import KpathSampler


warnings.filterwarnings("ignore")


class KpathSamplerTest(unittest.TestCase):
    def test_get_kpoints(self):
        pass


    def test_get_kpath(self):
        ### Part I. Get DStructure from atom.config
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/kpath/atom.config"
        file_path = "/data/home/liuhanyu/hyliu/pwmat_demo/scf_3d/atom.config"
        
        coords_are_cartesian = False
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        
        ### Part II. Setting torlenrance
        dimension = 3
        symprec = 0.1
        angle_tolerance = 5
        atol = 1e-5
        density = 0.01

        ### Part III. test
        kpath_sampler = KpathSampler(
                            structure=structure,
                            dimension=dimension,
                            symprec=symprec,
                            angle_tolerance=angle_tolerance,
                            atol=atol,
                            )
        kpath_sampler.HIGHK_file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/kpath/HIGHK"
        kpath_sampler.gen_kpt_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/kpath/gen.kpt"

        kpath_sampler.HIGHK_file_path = "/data/home/liuhanyu/hyliu/pwmat_demo/scf_3d/HIGHK"
        kpath_sampler.gen_kpt_path = "/data/home/liuhanyu/hyliu/pwmat_demo/scf_3d/gen.kpt"
               
        print(kpath_sampler.kpoints)
        print(kpath_sampler.kpaths)
        kpath_sampler.output_HIGHK_file()
        kpath_sampler.output_gen_kpt(density=density)



if __name__ == "__main__":
    unittest.main()