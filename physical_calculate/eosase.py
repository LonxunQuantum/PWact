#!/usr/bin/env python3
import sys
import ase
import ase.eos
import ase.units
import numpy as np
filename = 'summary'
# with open(filename) as fin:
#     txt = fin.readlines()
#     #fin.close()
# num_energy = len(txt)
# volumes = [0.0 for i in range(num_energy)]
# energies = [0.0 for i in range(num_energy)]
# for i in range(num_energy):
#     volumes[i] = float(txt[i].split()[0])# ** 3.0
#     energies[i] = float(txt[i].split()[1])
a = np.loadtxt(filename)
volumes = a[:,0]
energies = a[:,1]
EOS = ase.eos.EquationOfState(volumes,energies)
v0, e0, B = EOS.fit()
print('v0, a0_if_cubic: ', v0, v0 ** (1.0/3.0))
print('B: %E GPa' % (B / ase.units.kJ * 1.0e24))  # 1kJ=6.241509125883258 E+21 eV
print('e0: ', e0)
if len(sys.argv) > 1:
    print('no BM.png output')
    sys.exit(0)

EOS.plot('BM.png')
