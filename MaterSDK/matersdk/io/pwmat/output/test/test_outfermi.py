import unittest

# python3 -m matersdk.io.pwmat.output.test.test_outfermi
from ..outfermi import OutFermi


class OutFermiTest(unittest.TestCase):
    def test_out_fermi(self):
        out_fermi_path = "/data/home/liuhanyu/hyliu/pwmat_demo/dos/OUT.FERMI"
        
        out_fermi = OutFermi(out_fermi_path=out_fermi_path)
        
        print("\n1. 费米能级(unit: eV)为:", end="\t")
        print(out_fermi.get_efermi())
    

if __name__ == "__main__":
    unittest.main()