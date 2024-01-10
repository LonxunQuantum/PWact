import unittest

# python3 -m matersdk.io.vasp.output.test.test_outcar
from ..outcar import DOutcar

outcar_path = "/data/home/liuhanyu/hyliu/vasp_demo/ReNbSSe/0/OUTCAR"


class OutcarTest(unittest.TestCase):
    def test_all(self):
        doutcar = DOutcar(file_path=outcar_path)
        print("\nStep 1.")
        print("Step 1.1. free energy    TOTEN = ", doutcar.get_fr_energy())
        print("Step 1.2. energy(sigma->0) = ", doutcar.get_energy())


if __name__ == "__main__":
    unittest.main()