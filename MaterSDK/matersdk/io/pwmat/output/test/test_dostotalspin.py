import unittest

# python3 -m matersdk.io.pwmat.output.test.test_dostotalspin
from ..dostotalspin import Dostotalspin

class DostotalspinTest(unittest.TestCase):
    def test_dostotalspin(self):
        dos_totalspin_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/dos/DOS.totalspin_projected"
        dos_totalspin = Dostotalspin(dos_totalspin_path=dos_totalspin_path)
        
        print("\n1. tdos:")
        print(dos_totalspin.get_tdos())
        
        print("\n2. pdos on elements:")
        print(dos_totalspin.get_pdos_elements())

        print("\n3. pdos on orbitals:")
        print(dos_totalspin.get_pdos_orbitals())

if __name__ == "__main__":
    unittest.main()