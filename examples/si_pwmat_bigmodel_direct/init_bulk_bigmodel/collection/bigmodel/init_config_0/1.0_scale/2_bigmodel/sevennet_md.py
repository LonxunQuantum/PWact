from ase.io import read
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.optimize import LBFGS
import numpy as np
from sevenn.sevennet_calculator import SevenNetCalculator

traj_name = "tmp.traj"
xyz_name = "traj.xyz"

calc = SevenNetCalculator()
fmax = 0.1
run_step = 10000

T = 300
P = 1.01325
P_in_ev_per_ang3 = P / 1602176.6208
atoms = read("POSCAR")
atoms.calc = calc
opt = LBFGS(atoms)

def atoms2xyzstr(atoms):
    num_atom = atoms.get_global_number_of_atoms()
    vol = atoms.get_volume()
    pos = atoms.positions
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    cell = atoms.cell
    virial = -atoms.get_stress(voigt=False) * vol
    xyzstr = "%d\n" % num_atom
    xyz_head = 'Lattice="%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" Properties=species:S:1:pos:R:3:forces:R:3 energy=%.8f' 
    xyz_format = (cell[0,0],cell[0,1],cell[0,2],cell[1,0],cell[1,1],cell[1,2],cell[2,0],cell[2,1],cell[2,2],energy)
    if virial is not None:
        xyz_head += ' virial="%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f"'
        xyz_format += (
            virial[0,0], virial[0,1], virial[0,2],
            virial[1,0], virial[1,1], virial[1,2],
            virial[2,0], virial[2,1], virial[2,2]
            )
    xyz_head += '\n'
    xyzstr += xyz_head % xyz_format
    for i in range(num_atom):
        xyzstr += "%2s %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n" %\
        (atoms[i].symbol,pos[i,0],pos[i,1],pos[i,2],forces[i,0],forces[i,1],forces[i,2])
    return xyzstr

def domd():
    opt.run(fmax=fmax)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    '''
    print("Beginning Nose-Hoover NVT equilibration")
    dyn_nvt = NPT(
    atoms=atoms,
    timestep=.5*fs,
    temperature_K=600,
    externalstress=P_in_ev_per_ang3,
    ttime=25*fs,
    pfactor=None,
    logfile="nvt.log",
    loginterval=200
    )
    dyn_nvt.run(200000)
    '''

    print("Beginning Nose-Hoover NPT equilibration")
    dyn_npt = NPT(
    atoms=atoms,
    timestep=.5*fs,
    temperature_K=T,
    externalstress=P_in_ev_per_ang3,
    mask=np.eye(3),
    ttime=25*fs,
    pfactor=50*fs,
    trajectory=traj_name,
    logfile="npt.log",
    loginterval=100
    )
    dyn_npt.run(run_step)

def dolabel():
    traj = read(traj_name, index=":")
    output_file = xyz_name
    f = open(output_file, "w")
    for i in range(len(traj)):
        atoms = traj[i]
        atoms.calc = calc
        f.write(atoms2xyzstr(atoms))
    f.close()

if __name__=="__main__":
    domd()
    dolabel()