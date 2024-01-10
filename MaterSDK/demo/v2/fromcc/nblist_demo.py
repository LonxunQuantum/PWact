from pymatgen.core import Structure
import numpy as np
import time
import copy


# lattice, frac_coords still lose accuracy due to unclear reasons.
poscar_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf_/POSCAR"
structure = Structure.from_file(poscar_path)    #.make_supercell([3, 4, 1])
lattice = structure.lattice.matrix
lattice = copy.deepcopy(lattice)
for ii in range(3):
    for jj in range(3):
        print(lattice[ii][jj], end=", ")
    print("\n")
species = np.array([tmp_specie.Z for tmp_specie in structure.species])
species = np.where(species==42, 0, 1)
frac_coords = structure.frac_coords
rcut = 3.3
rcut_smooth = 3.0
pbc_xyz = [True, True, False]
umax_num_neigh_atoms = 19
umax_num_neigh_atoms_lst = np.array([10, 9]).astype(np.int32)
sort = True


from matersdk.fromcc import nblist

start = time.time()    
info = nblist.find_info4mlff(
    lattice.astype(np.float64),
    species.astype(np.int32),
    frac_coords.astype(np.float64),
    rcut,
    pbc_xyz,
    umax_num_neigh_atoms,
    sort
)
end = time.time()
#print("Consuming time in seconds: ", end-start)

inum:int = info[0]
ilist:np.array = info[1]
numneigh:np.array = info[2]
firstneigh:np.array = info[3]
relative_coords:np.array = info[4]
types:np.array = info[5]
nghost:int = info[6]


"""
for ii in range(12):
    for jj in range(19):
        tmp_rc = info[4][ii][jj]
        print(np.linalg.norm(tmp_rc), end=", ")
    print("\n")
"""



import torch
from matersdk.fromcc import envMatrixOp
# 1.
ilist_tensor:torch.tensor = torch.from_numpy(ilist)
ilist_tensor.unsqueeze_(0)
# 2. 
numneigh_tensor:torch.tensor = torch.from_numpy(numneigh)  
numneigh_tensor.unsqueeze_(0)
# 3.
firstneigh_tensor:torch.tensor = torch.from_numpy(firstneigh)
firstneigh_tensor.unsqueeze_(0)
# 4.
relative_coords_tensor:torch.tensor = torch.from_numpy(relative_coords)
relative_coords_tensor = relative_coords_tensor.to(torch.float32)
relative_coords_tensor.unsqueeze_(0)
# 5.
types_tensor:torch.tensor = torch.from_numpy(types)
types_tensor.unsqueeze_(0)
# 6. 
umax_num_neigh_atoms_lst_tensor:torch.tensor = torch.from_numpy(umax_num_neigh_atoms_lst)
umax_num_neigh_atoms_lst_tensor.unsqueeze_(0)


time1 = time.time()
for _ in range(300):
    tilde_r, tilde_r_deriv = envMatrixOp(
        ilist_tensor,
        numneigh_tensor,
        firstneigh_tensor,
        relative_coords_tensor,
        types_tensor,
        umax_num_neigh_atoms_lst_tensor,
        rcut,
        rcut_smooth)
time2 = time.time()  
print("Consuming time in seconds: ", time2 - time1)
print(tilde_r.size())
print(tilde_r_deriv.size())
