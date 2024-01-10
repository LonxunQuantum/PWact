import unittest


# python3 -m matersdk.io.publicLayer.test.test_molecule
from ..molecule import DMolecule
from ..structure import DStructure


class DMoleculeTest(unittest.TestCase):
    def test_all(self):
        pdb_path = "/data/home/liuhanyu/hyliu/pwmat_demo/fmt/atom01.pdb"
        json_path = ""
        molecule = DMolecule.from_file(
                        file_path=pdb_path,
                        file_format="pdb")
        print(molecule)
        #molecule.to(filename=None, fmt="json")
        
        #structure = DStructure.from_file(
        #                file_path=None,
        #                file_format="json")
        #print(structure)



if __name__ == "__main__":
    unittest.main()