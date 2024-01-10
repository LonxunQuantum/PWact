import unittest

# python3 -m matersdk.io.pwmat.utils.test.test_acextractor
from ..acextractor import ACExtractor


class ACExtractorTest(unittest.TestCase):
    def test_all(self):
        atom_config_path = "/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/SiC/atom.config"
        atom_config_path = "/data/home/liuhanyu/hyliu/pwmat_demo/CrI3/scf/test.config"
        
        ace = ACExtractor(file_path=atom_config_path)
        ### Step 1. 
        print()
        num_atoms = ace.get_num_atoms()
        print("1. Number of atoms in structure = {0}".format(num_atoms))
        
        ### Step 2. 
        print()
        basis_vectors = ace.get_basis_vectors()
        print("2. Basis vectors in structure = ")
        print(basis_vectors)
        
        ### Step 3.
        print()
        types = ace.get_types()
        print("3. Types in structure = ")
        print(types)

        ### Step 4.
        print()
        coords = ace.get_coords()
        print("4. Frac coords in structure = ")
        print(coords)


        ### Step 5.
        print()
        magmoms = ace.get_magmoms()
        print("5. Magmoms in structure = ")
        print(magmoms)


if __name__ == "__main__":
    unittest.main()