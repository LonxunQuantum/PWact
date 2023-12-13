#!/usr/bin/env python3
# image class, again, hahaha
# all forces in python variables are correct one f_atom = -dE/dR
# please add minus sign when you read and write MOVEMENT
import subprocess
from shutil import move, copy
import numpy as np
import numpy.linalg as LA
import dpdata
import os
import argparse

from pyparsing import line_end

element = ["0", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg"]

class Image():
    def __init__(self, num_atoms, lattice, type_atom, x_atom, f_atom, e_atom, ddde, e_potential):
        self.num_atoms = num_atoms
        self.lattice = lattice
        self.type_atom = type_atom
        self.x_atom = x_atom
        self.f_atom = f_atom
        self.e_atom = e_atom
        self.e_potential = e_potential
        self.ddde = ddde

        self.egroup = np.zeros((num_atoms), dtype=float)
        
    # This member function won't be used in modifying MOVEMENT
    def calc_egroup(self):
        # to line 40, get e_atom0
        f = open(r'fread_dfeat/feat.info', 'r')
        txt = f.readlines()
        f.close()
        iflag_pca = int(txt[0].split()[0])
        num_feat_type = int(txt[1].split()[0])
        for i in range(num_feat_type):
            ifeat_type = int(txt[2+i].split()[0])
        num_atomtype = int(txt[2+num_feat_type].split()[0].split(',')[0])
        itype_atom = np.zeros((num_atomtype), dtype=int)
        nfeat1 = np.zeros((num_atomtype), dtype=int)
        nfeat2 = np.zeros((num_atomtype), dtype=int)
        nfeat2_integral = np.zeros((num_atomtype), dtype=int)
        nfeat = np.zeros((num_feat_type, num_atomtype), dtype=int)
        ipos_feat = np.zeros((num_feat_type, num_atomtype), dtype=int)
        for i in range(num_atomtype):
            tmp = [int(kk) for kk in txt[3+num_feat_type+i].split()]
            itype_atom[i] = tmp[0]
            nfeat1[i] = tmp[1]
            nfeat2[i] = tmp[2]
        for i in range(num_atomtype):
            nfeat2_integral[i] = np.sum(nfeat2[0:i+1])

        # read fit_linearMM.input
        f = open(r'fread_dfeat/fit_linearMM.input', 'r')
        txt = f.readlines()
        f.close()
        tmp_ntype = int(txt[0].split()[0].split(',')[0])
        tmp_m_neigh = int(txt[0].split()[1].split(',')[0])
        type_map = [ 0 for i in range(tmp_ntype)]
        for i in range(tmp_ntype):
            type_map[i] = int(txt[1+i].split()[0].split(',')[0])

        dwidth = float(txt[tmp_ntype+2].split()[0])

        # read linear_fitB.ntype
        f = open(r'fread_dfeat/linear_fitB.ntype', 'r')
        txt = f.readlines()
        f.close()
        e_atom0 = np.zeros((num_atomtype), dtype=float)
        for i in range(num_atomtype):
            e_atom0[i] = float(txt[nfeat2_integral[i]].split()[1])

        # calc distance
        # num_atoms, x_atom, lattice
        distance_matrix = np.zeros((self.num_atoms, self.num_atoms), dtype=float)
        for i in range(self.num_atoms):
            d0 = self.x_atom - self.x_atom[i]
            d1 = np.where(d0<-0.5, d0+1.0, d0)
            dd = np.where(d1>0.5, d1-1.0, d1)
            d_cart = np.matmul(dd, self.lattice)
            distance_matrix[i] = np.array([ LA.norm(kk) for kk in d_cart])

        fact_matrix = np.exp(-distance_matrix**2/dwidth**2)
        e_atom0_array = np.zeros((self.num_atoms), dtype=float)
        for i in range(self.num_atoms):
            e_atom0_array[i] = e_atom0[type_map.index(self.type_atom[i])]

        for i in range(self.num_atoms):
            esum1 = ((self.e_atom - e_atom0_array)*fact_matrix[i]).sum()
            self.egroup[i] = esum1 / fact_matrix[i].sum()

def write_image(fout, image):
    fout.write(" %d atoms, Iteration (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E\n"\
                % (image.num_atoms, 0.0, image.e_potential, image.e_potential, 0.0))
    fout.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
    fout.write("          1    0.5000000000E+00   0.59978E+03   0.30000E+03   0.59978E+03   0.50000E+02   0.59978E+03\n")
    fout.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
    fout.write("          -0.1971547257E+05\n")
    fout.write("Lattice vector (Angstrom)\n")
    for i in range(3):
        fout.write("  %16.10E    %16.10E    %16.10E\n" % (image.lattice[i][0], image.lattice[i][1], image.lattice[i][2]))
    fout.write("  Position (normalized), move_x, move_y, move_z\n")
    for i in range(image.num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                 % (image.type_atom[i], image.x_atom[i][0], image.x_atom[i][1], image.x_atom[i][2]))
    fout.write("  Force (-force, eV/Angstrom)\n")
    for i in range(image.num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                 % (image.type_atom[i], -image.f_atom[i][0], -image.f_atom[i][1], -image.f_atom[i][2]))  # minus sign here
    fout.write("  Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  %20.15E\n " % image.ddde)
    for i in range(image.num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                 % (image.type_atom[i], image.e_atom[i], 0.0, 0.0))
    fout.write(' -------------------------------------\n')
    # idtk = i don't know


def outcar2raw():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='OUTCAR')
    parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=2000)
    parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='.')
    args = parser.parse_args()

    dpdata.LabeledSystem(args.input, fmt='vasp/outcar').to('deepmd/raw', args.directory)

def outcar2raw(outcar_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default=outcar_path)
    parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=2000)
    parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='.')
    args = parser.parse_args()

    dpdata.LabeledSystem(args.input, fmt='vasp/outcar').to('deepmd/raw', args.directory)

"""
@Description :
输入outcar上级的上级文件目录、数据保存目录
将vasp outcar文件转为movement文件
@Returns     :
@Author       :wuxingxing
"""

def dpmd2movement(outcar_dir, movement_dir):
    movement_list = []
    movement_index = 0
    outcar_dir_list = sorted(os.listdir(outcar_dir), key=lambda x: os.path.getmtime(os.path.join(outcar_dir, x)))

    for i in outcar_dir_list:
        i_path = os.path.join(outcar_dir, i)
        if os.path.isdir(i_path) is False:
            continue
        for j in os.listdir(i_path):
            if "OUTCAR" in j:
                outcar_path = os.path.join(i_path, "OUTCAR")
                dpdata_path = os.path.join(movement_dir, "dpraw/{}/dpmd_raw".format(movement_index))
                dpdata.LabeledSystem(outcar_path).to('deepmd/raw', dpdata_path)
                raw_path = os.path.join(dpdata_path)
                type_raw = np.loadtxt(r"{}/type.raw".format(raw_path), dtype=int)
                box_raw = np.loadtxt(r"{}/box.raw".format(raw_path))
                coord_raw = np.loadtxt(r"{}/coord.raw".format(raw_path))
                energy_raw = np.loadtxt(r"{}/energy.raw".format(raw_path))
                force_raw = np.loadtxt(r"{}/force.raw".format(raw_path))
                fin = open(r"{}/type_map.raw".format(raw_path))
                type_map_txt = fin.readlines()
                fin.close()

                print ("raw data reading completed\n")	
                num_type = len(type_map_txt)
                num_atom = type_raw.shape[0]
                num_image = coord_raw.shape[0]
                type_map = [ 'H' for tmp in range(num_type)]
                type_atom = np.zeros((num_atom), dtype=int)

                for i in range(num_type):
                    type_map[i] = type_map_txt[i].split()[0]
                for i in range(num_atom):
                    type_atom[i] = element.index(type_map[type_raw[i]])

                all_images = []
                for i in range(num_image):
                    lattice = box_raw[i].reshape(3,3)
                    x_atom = np.dot(coord_raw[i].reshape(num_atom,3),LA.inv(lattice))
                    f_atom = force_raw[i].reshape(num_atom,3)
                    tmp_eatom = energy_raw[i] / num_atom
                    e_atom = np.array([tmp_eatom for i in range(num_atom)])
                    ddde = 0.0
                    e_potential = energy_raw[i]
                    tmp_image = Image(num_atom, lattice, type_atom, x_atom, f_atom, e_atom, ddde, e_potential)
                    all_images.append(tmp_image)

                movement_path = os.path.join(movement_dir, "PWdata/{}".format(movement_index))
                if os.path.exists(movement_path) is False:
                    os.makedirs(movement_path)
                movement_index += 1
                fout = open('{}/MOVEMENT'.format(movement_path), 'w')
                for i in range(num_image):
                    write_image(fout, all_images[i])
                fout.close()
    #删除dpdata path, this path is only used to generate movement files
    # import shutil
    # shutil.rmtree(os.path.join(movement_dir, "dpraw"))

"""
@Description :
movement convert to dpkf training data 
@Returns     :
@Author       :wuxingxing
"""
def movement2traindata(work_dir, parameter_path):
    def make_dpkf_data(bash_path):
        res = "\n"
        res += "#!/bin/bash -l\n"
        res += "conda activate mlff_env\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "mlff.py 1>> mlff_run.log 2>>mlff_error.log\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "seper.py 1>> seper_run.log 2>>seper_error.log\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "gen_dpdata.py 1>> gen_dpdata_run.log 2>>gen_dpdata_error.log\n"
        res += "test $? -ne 0 && exit 1\n\n"
        res += "echo 0 > gen_dpkf_data_success.tag\n\n"
        res += "rm *.log\n"
        res += "test $? -ne 0 && exit 1\n"
        with open(bash_path, "w") as wf:
            wf.write(res)

    if os.path.exists(os.path.join(work_dir, "train/davg.npy")) == True:
        return

    cwd = os.getcwd()
    os.chdir(work_dir)
    if os.path.exists("parameters.py") is False:
        copy(parameter_path, "parameters.py")
    print("copy parameter done")
    
    make_dpkf_data(os.path.join(work_dir, "gen_dpkf_data.sh"))
    print("make gen_dpkf_data.sh done")
    result = subprocess.call("bash -i gen_dpkf_data.sh", shell=True)
    print("bash -i gen_dpkf_data.sh run result: {}".format(result))
    assert(os.path.exists("train_data/final_train/natoms_img.npy") == True)
    print("make_dpkf_data done")

    os.chdir(cwd)


"""
@Description :
when the energy and force of a atom.config is calculate by PWmat scf methond,
pwmat will give a OUT.ENDIV file, OUT.FORCE, we need to use these info to construct
a movement file base on the atom.config file:
1. replace Force (-force, eV/Angstrom) of atom.config
2. read Force (-force, eV/Angstrom) from OUT.FORCE
3. read Atomic-Energy from OUT.ENDIV
4. read Ep from OUT.MLMD
5. replace Ep, Force block and atom energy block

@Returns     :
@Author       :wuxingxing
"""

class Scf2Movement(object):
    def __init__(self, atom_config_path, out_force_path, atom_energy_path, mlmd_path, save_path):
        self.atom_config = self.read_file_as_lines(atom_config_path)
        self.atom_nums = int(self.atom_config[0].split()[0])
        self.out_force = self.read_file_as_lines(out_force_path)
        self.atom_energy = self.read_file_as_lines(atom_energy_path)
        self.mlmd = self.read_file_as_lines(mlmd_path)
        self.save_path = save_path
        self.scp2movement()

    def scp2movement(self):
        self.replace_ep()
        self.replace_force_block(self.atom_nums)
        self.replace_atom_energy_block(self.atom_nums)
        self.save_movement()

    def read_file_as_lines(self, file_path):
        with open(file_path, 'r') as rf:
            lines = rf.readlines()
        return lines
    
    def replace_ep(self):
        if "Ep" not in self.atom_config[0]:
            self.atom_config.insert(0, self.mlmd[0])
        else:
            replace = self.mlmd[0]
            self.atom_config[0] = replace

    def replace_atom_energy_block(self, atom_nums):
        replace = self.atom_energy[2: atom_nums+2]
        i = 0
        while ("Atomic-Energy" not in self.atom_config[i]):
            i += 1
            if i >= len(self.atom_config):
                break
        if i >= len(self.atom_config):
            self.atom_config.insert(i, "Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=   0.0000000000E+00\n")
            i = i+1
            for rep in replace:
                self.atom_config.insert(i, rep)
                i =i+1
        else:
            i += 1
            self.atom_config[i:i+len(replace)] = replace

    def replace_force_block(self, atom_nums):
        # read OUT.FORCE force (eV/A) block
        replace = self.out_force[1: atom_nums+1]

        #  replace Velocity  (bohr/fs) block
        i = 0
        while ("Force (-force, eV/Angstrom)" not in self.atom_config[i]):
            i += 1
            if i >= len(self.atom_config):
                break
        if i >= len(self.atom_config):
            self.atom_config.insert(i,"Force (-force, eV/Angstrom)\n") #如果不存在该块，则新建一个，用于v0计算
            i = i+1
            for rep in replace:
                self.atom_config.insert(i, rep)
                i =i+1
        else:
            i = i + 1 # current line is "Force (-force, eV/Angstrom)"
            self.atom_config[i:i+len(replace)] = replace
        
    def save_movement(self):
        with open(self.save_path, 'w') as wf:
            for i in self.atom_config:
                wf.write(i)

if __name__ == '__main__':
# #     # 'OUTCAR' to MOVEMENT
# #     outcars_dir = "CH4.POSCAR.01x01x01/02.md/sys-0004-0001/scale-1.000"
#     movement_dir = "/home/wuxingxing/datas/active_learning_dir/GaAs_system/iter.0000/exploring/md_dpkf_dir"
# # #     dpmd2movement(outcars_dir, movement_dir)

#     # MOVEMENT to dpkf train-data
#     parameter_path = "/home/wuxingxing/datas/active_learning_dir/GaAs_system/init_data/GaAs_init_train_dp/parameters.py"
#     movement2traindata(movement_dir, parameter_path)
    comm_path = '/share/home/wuxingxing/al_dir/cu_system/test/makemovement'
    atom_config_path = os.path.join(comm_path, "atom.config")
    out_force_path = os.path.join(comm_path, "OUT.FORCE")
    atom_energy_path = os.path.join(comm_path, "OUT.ENDIV")
    mlmd_path = os.path.join(comm_path, "OUT.MLMD")
    save_path =  os.path.join(comm_path, "MOVEMENT")
    am = Scf2Movement(atom_config_path, out_force_path, atom_energy_path, mlmd_path, save_path)