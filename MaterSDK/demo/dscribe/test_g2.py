from ase.io import read as ase_read
from ase.build import make_supercell
from dscribe.descriptors import ACSF
from timeit import default_timer as timer
import os

os.chdir()
structure = ase_read(filename="/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/POSCAR")
structure.set_pbc((True, True, False))
supercell = make_supercell(structure, [[10, 0, 0], [0, 1, 0], [0, 0, 1]])
num_atoms = len(supercell)
print( len(supercell) )

io.write('expanded_cell.poscar', supercell)


start = timer()
acsf = ACSF(
    species=["Mo", "S"],
    r_cut=3.3,
    g2_params=[[1, 1]])


acsf_value = acsf.create(supercell, centers=[*range(num_atoms)])
#acsf_deriv = acsf.create(structure, centers)
end = timer()

print("Costing time: ", (end - start) * 10E6)