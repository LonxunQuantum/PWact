"""
    Data pre-processing for LAMMPS
    L. Wang
    2023.1
"""

import numpy as np
from utils.constant import ELEMENTTABLE, ELEMENTMASSTABLE

def normalize(v):
  norm = np.linalg.norm(v)
  if norm == 0:
    return [0, 0, 0]
  return v/norm

def pBox2l(lattice):
    
  from numpy.linalg import norm
  from numpy import dot
  from numpy import cross
  """
      converting PWMAT box to lammps style upper-triangle
  """

  A = lattice[0]
  B = lattice[1]
  C = lattice[2]


  nA = normalize(A)
  Al = np.linalg.norm(A)
  Bl = np.linalg.norm(B)
  Cl = np.linalg.norm(C)

  ax = np.linalg.norm(A)
  bx = np.dot(B,nA)
  by = np.sqrt(Bl*Bl-bx*bx)
  cx = np.dot(C,nA)
  cy = (np.dot(B,C)-bx*cx)/by
  cz = np.sqrt(Cl*Cl - cx*cx - cy*cy)

  xx = ax
  yy = by
  zz = cz
  xy = bx
  xz = cx
  yz = cy 

  return [xx,xy,yy,xz,yz,zz]
    
def p2l(filename = "POSCAR", output_name = "lammps.data"):
    """
        poscar to lammps.data
        
        NOTE: in PWMAT, each ROW represnets a edge vector
    """
    natoms = 0
    atype = []
    
    A = np.zeros([3,3],dtype=float)
    
    infile = open(filename, 'r')
    
    raw = infile.readlines()
    raw = raw[2:]
    
    infile.close()
    
    for idx, line in enumerate(raw):
        raw[idx] = line.split() 
    
    # pwmat box
    for i in range(3):
        #print (raw[i])
        A[i,0] = float(raw[i][0])
        A[i,1] = float(raw[i][1])
        A[i,2] = float(raw[i][2])
        
    lammps_box = pBox2l(A)

    # number of type
    typeNum = len(raw[4])
    
    p = 1 
    # atom num & type list 
    for item in raw[4]:
        natoms += int(item)
        atype += [p for i in range(int(item))]
        p +=1
    
    # x array 
    x = np.zeros([natoms,3],dtype=float)
    
    if "SELECT" in raw[5][0].upper():
        n_pos = 7
    else:
        n_pos = 6

    for idx,line in enumerate(raw[n_pos:]):
        
        x[idx,0] = float(line[0])
        x[idx,1] = float(line[1])
        x[idx,2] = float(line[2])
    
        
    # output preparation 
    # return [xx,xy,yy,xz,yz,zz]
    xlo = 0.0
    xhi = lammps_box[0]
    ylo = 0.0
    yhi = lammps_box[2]
    zlo = 0.0
    zhi = lammps_box[5]
    xy = lammps_box[1]
    xz = lammps_box[3]
    yz = lammps_box[4]

    LX = np.zeros([natoms,3],dtype=float)
    A = np.zeros([3,3],dtype=float) 
    
    A[0,0] = lammps_box[0]
    A[0,1] = lammps_box[1]
    A[1,1] = lammps_box[2]
    A[0,2] = lammps_box[3]
    A[1,2] = lammps_box[4]
    A[2,2] = lammps_box[5]
    
    print("converted LAMMPS upper trangualr box:")
    
    print(A)
    
    print("Ref:https://docs.lammps.org/Howto_triclinic.html")
    # convert lamda (fraction) coords x to box coords LX
    # A.T x = LX
    # LX = A*x in LAMMPS. see https://docs.lammps.org/Howto_triclinic.html
    """
    for i in range(natoms):
        LX[i,0] = A[0,0]*x[i,0] + A[1,0]*x[i,1] + A[2,0]*x[i,2]
        LX[i,1] = A[0,1]*x[i,0] + A[1,1]*x[i,1] + A[2,1]*x[i,2]
        LX[i,2] = A[0,2]*x[i,0] + A[1,2]*x[i,1] + A[2,2]*x[i,2]
    """
    
    for i in range(natoms):
        LX[i,0] = A[0,0]*x[i,0] + A[0,1]*x[i,1] + A[0,2]*x[i,2]
        LX[i,1] = A[1,0]*x[i,0] + A[1,1]*x[i,1] + A[1,2]*x[i,2]
        LX[i,2] = A[2,0]*x[i,0] + A[2,1]*x[i,1] + A[2,2]*x[i,2]

    #print(A)
    #AI = np.linalg.inv(A)
    #print(AI)

    # output LAMMPS data
    ofile = open(output_name, 'w')

    ofile.write("#converted from POSCAR\n\n")

    ofile.write("%-12d atoms\n" % (natoms))
    ofile.write("%-12d atom types\n\n" % (typeNum))

    ofile.write("%16.12f %16.12f xlo xhi\n" %  (xlo, xhi))
    ofile.write("%16.12f %16.12f ylo yhi\n" %  (ylo, yhi))
    ofile.write("%16.12f %16.12f zlo zhi\n" %  (zlo, zhi))  
    ofile.write("%16.12f %16.12f %16.12f xy xz yz\n\n" %  (xy, xz, yz))

    ofile.write("Masses\n\n")

    for idx,sym in enumerate(raw[3]):
        out_line = str(idx+1)+" "
        out_line += str(ELEMENTMASSTABLE[ELEMENTTABLE[sym]])+"\n"
        #print (out_line)
        ofile.write(out_line)
        
    #ofile.write("1 6.94000000      #Li\n")
    #ofile.write("2 180.94788000    #Ta\n")
    #.write("3 15.99900000     #O\n")
    #ofile.write("4 1.00800000      #H\n\n")

    ofile.write("\nAtoms # atomic\n\n")

    for i in range(natoms):
        ofile.write("%12d %5d %21.15f %21.15f %21.15f\n" % (i+1, atype[i], LX[i,0], LX[i,1], LX[i,2]) )

    ofile.close()

if __name__ =="__main__":
    
    p2l() 