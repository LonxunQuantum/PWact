import unittest

# python3 -m matersdk.adalearn.generator.test.test_perturbation
from ..perturbation import (
                    PerturbStructure, 
                    BatchPerturbStructure
)


class BatchPerturbStructureTest(unittest.TestCase):
    def test_all(self):
        Perturbed = ['/data/home/liuhanyu/hyliu/code/test']
        pert_num = 50
        cell_pert_fraction = 0.03
        atom_pert_distance = 0.01
        BatchPerturbStructure.batch_perturb(
            Perturbed=Perturbed,
            pert_num=pert_num,
            cell_pert_fraction=cell_pert_fraction,
            atom_pert_distance=atom_pert_distance
        )


if __name__ == "__main__":
    unittest.main()