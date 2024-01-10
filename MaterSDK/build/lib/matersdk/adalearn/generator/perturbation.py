import numpy as np
from typing import List
import pathlib
import os, sys

#codepath = str(pathlib.Path(__file__).parent.resolve())
#sys.path.append(codepath)
#sys.path.append(codepath+'/MaterSDK')

from matersdk.io.publicLayer.structure import DStructure


class PerturbStructure(object):
    def __init__(self, atoms:List[DStructure]):
        self.atoms = atoms


    def perturb(self,
        pert_num:int,
        cell_pert_fraction:float,
        atom_pert_distance:float):
        """
        Perturb each frame in the system randomly.
        The cell will be deformed randomly, and atoms will be displaced by a random distance in random direction.

        Parameters
        ----------
        pert_num : int
            Each frame in the system will make `pert_num` copies,
            and all the copies will be perturbed.
            That means the system to be returned will contain `pert_num` * frame_num of the input system.
        cell_pert_fraction : float
            A fraction determines how much (relatively) will cell deform.
            The cell of each frame is deformed by a symmetric matrix perturbed from identity.
            The perturbation to the diagonal part is subject to a uniform distribution in [-cell_pert_fraction, cell_pert_fraction),
            and the perturbation to the off-diagonal part is subject to a uniform distribution in [-0.5*cell_pert_fraction, 0.5*cell_pert_fraction).
        atom_pert_distance: float
            unit: Angstrom. A distance determines how far atoms will move.
            Atoms will move about `atom_pert_distance` in random direction.
            The distribution of the distance atoms move is determined by atom_pert_style

        Returns
        -------
        perturbed_system : list of ase Atoms
            The perturbed structs. It contains `pert_num` * frame_num of the input system frames.
        """
        perturbed_structs = []
        for jj in range(pert_num):
            tmp_system = self.atoms.copy()
            cell_perturb_matrix = self.get_cell_perturb_matrix(cell_pert_fraction)
            tmp_system.cell = np.matmul(tmp_system.lattice.matrix, cell_perturb_matrix)
            tmp_system.coords = np.matmul(tmp_system.cart_coords, cell_perturb_matrix)
            for kk in range(len(tmp_system.coords)):
                atom_perturb_vector = self.get_atom_perturb_vector(atom_pert_distance)
                tmp_system.coords[kk] += atom_perturb_vector
            new_system = self.rot_lower_triangular(tmp_system)
            perturbed_structs.append(new_system)
        return perturbed_structs


    def get_cell_perturb_matrix(self, cell_pert_fraction):
        if cell_pert_fraction<0:
            raise RuntimeError('cell_pert_fraction can not be negative')
        e0 = np.random.rand(6)
        e = (e0 * 2 - 1)* cell_pert_fraction
        cell_pert_matrix = np.array(
            [[1+e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], 1+e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], 1+e[2]]]
        )
        return cell_pert_matrix


    def get_atom_perturb_vector(self, atom_pert_distance):
        random_vector = None
        if atom_pert_distance < 0:
            raise RuntimeError('atom_pert_distance can not be negative')

        e = np.random.randn(3)
        random_vector = atom_pert_distance * e / np.sqrt(3)

        return random_vector


    def rot_lower_triangular(self, atoms) :
        qq, rr = np.linalg.qr(atoms.cell.T)
        if np.linalg.det(qq) < 0 :
            qq = -qq
            rr = -rr
        atoms.cell = np.matmul(atoms.cell, qq)
        atoms.coords = np.matmul(atoms.coords, qq)
        rot = np.eye(3)
        if atoms.cell[0][0] < 0 :
            rot[0][0] = -1
        if atoms.cell[1][1] < 0 :
            rot[1][1] = -1
        if atoms.cell[2][2] < 0 :
            rot[2][2] = -1
        assert(np.linalg.det(rot) == 1)
        new_system = DStructure(
                lattice=np.matmul(atoms.cell, rot),
                species=atoms.species,
                coords=np.matmul(atoms.coords, rot),
                coords_are_cartesian=True
            )
        # atoms.lattice.matrix = np.matmul(atoms.lattice.matrix, rot)
        # atoms.cart_coords = np.matmul(atoms.cart_coords, rot)
        return new_system



class BatchPerturbStructure(object):
    @staticmethod
    def batch_perturb(
            Perturbed:List[str],
            pert_num:int,
            cell_pert_fraction:float,
            atom_pert_distance:float):
        for tmp_system in Perturbed:
            tmp_structure = DStructure.from_file(tmp_system + "/atom.config", "pwmat")
            perturbed_obj = PerturbStructure(tmp_structure)
            perturbed_structs = perturbed_obj.perturb(
                                    pert_num=pert_num, 
                                    cell_pert_fraction=cell_pert_fraction,
                                    atom_pert_distance=atom_pert_distance)
            if not os.path.exists(tmp_system+'/'+'structures'):
                os.mkdir( os.path.join(tmp_system, 'structures') )
            for tmp_perturbed_idx, tmp_pertubed_struct in enumerate(perturbed_structs):
                tmp_pertubed_struct.to(
                            output_file_path=os.path.join(tmp_system, "structures", "{0}.config".format(tmp_perturbed_idx)),
                            output_file_format="pwmat")


'''
if __name__ == "__main__":
    def main():
        Systems_2be_perturbed = ['111']
        t = 50
        l = t -1    # 49
        for system in Systems_2be_perturbed:
            atoms = DStructure.from_file(system + "/atom.config", "pwmat")
            perturbed_obj = PerturbStructure(atoms)
            perturbed_structs = perturbed_obj.perturb(50,0.03,0.01)
            if not os.path.exists("./"+system+'/'+'structures'):
                    os.mkdir("./"+system+'/'+'structures')
            for struct in perturbed_structs:
                struct.to('%s/structures/%d' % (system,(t-l)), output_file_format='pwmat')
                t += 1
    main()
'''