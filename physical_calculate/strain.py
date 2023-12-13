#!/usr/bin/env python
import numpy as np
import os
import sys
'''
#!/bin/bash
for i in {1..258}
do
    echo ${i}
    cd ${i}

    for j in 1 2
    do
        for k in 0.990 0.995 1.000 1.005 1.010
        do
            cp -ra 0 ${j}_${k}
            cd ${j}_${k}
            ../../strain.py ${j} ${k}
            \mv poscar_${j}_${k} POSCAR
            cd ..
        done
    done

    cd ..
    done
'''

class strain_lattice():
    def __init__(self):
        self.read_poscar()

    def read_poscar(self):
        fin = open('POSCAR', 'r')
        txt = fin.readlines()
        fin.close()
        self.zoom = float(txt[1].split()[0])
        
        self.lat = np.zeros((3,3))
        a1 = [float(i) for i in txt[2].split()]
        a2 = [float(i) for i in txt[3].split()]
        a3 = [float(i) for i in txt[4].split()]
    
        self.lat[0] = np.array(a1)
        self.lat[1] = np.array(a2)
        self.lat[2] = np.array(a3)
    
        self.element = txt[5].split()
        self.n_atom = [int(i) for i in txt[6].split()]
        self.n_total = sum(self.n_atom)
        self.x_atom = np.zeros((self.n_total,3))
        self.d_or_c = txt[7][0]
        for i in range(self.n_total):
            self.x_atom[i] = np.array([float(col) for col in txt[i+8].split()])
    


    def strain_all(self, direction_list, ratio):
        delta = ratio - 1.0
        trans_matrix = np.array([ [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] for i in range(6) ], dtype=float)
        trans_matrix[1-1][0][0] = delta
        trans_matrix[2-1][1][1] = delta
        trans_matrix[3-1][2][2] = delta

        trans_matrix[4-1][1][2] = delta
        trans_matrix[4-1][2][1] = delta
        trans_matrix[5-1][0][2] = delta
        trans_matrix[5-1][2][0] = delta
        trans_matrix[6-1][0][1] = delta
        trans_matrix[6-1][1][0] = delta

        trans_total = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0] ]
        for i in direction_list:
            trans_total += trans_matrix[i-1]

        lat_new = np.dot(self.lat, trans_total)
        self.lat = lat_new
        

        ''' old code
        if direction == 0:
            #(a11' a21')   (1+e   0) ( a11 a21)
            #(         ) = (       ) (        )
            #(a12' a22')   (0     e) ( a12 a22)
            #self.lat[0] *= ratio
            #lat1_paralell_to_lat0 = np.dot(self.lat[1], self.lat[0]) * self.lat[0] / np.dot(self.lat[0], self.lat[0])
            #lat1_perpendicular_to_lat0 = self.lat[1] - lat1_paralell_to_lat0
            #self.lat[1] = ratio * lat1_paralell_to_lat0 + lat1_perpendicular_to_lat0
            delta = ratio - 1.0
            trans_matrix = np.array([ [1+delta, 0, 0], [0, 1, 0], [0, 0, 1] ])
            lat_new = np.dot(self.lat, trans_matrix)
            self.lat = lat_new
        elif direction == 1:
            #(a11' a21')   (1     0) ( a11 a21)
            #(         ) = (       ) (        )
            #(a12' a22')   (0   1+e) ( a12 a22)
            #lat1_paralell_to_lat0 = np.dot(self.lat[1], self.lat[0]) * self.lat[0] / np.dot(self.lat[0], self.lat[0])
            ##lat1_perpendicular_to_lat0 = self.lat[1] - lat1_paralell_to_lat0
            #self.lat[1] = lat1_paralell_to_lat0 + ratio * lat1_perpendicular_to_lat0
            delta = ratio - 1.0
            trans_matrix = np.array([ [1, 0, 0], [0, 1+delta, 0], [0, 0, 1] ])
            lat_new = np.dot(self.lat, trans_matrix)
            self.lat = lat_new
        elif direction == 6:
            # strain along bisector of x axis and y axis, 45 degree
            #(a11' a21')   (1  e) ( a11 a21)
            #(         ) = (    ) (        )
            #(a12' a22')   (e  1) ( a12 a22)
            delta = ratio - 1.0
            trans_matrix = np.array([ [1+2*delta, 1, 0], [0, 1+2*delta, 0], [0, 0, 1] ])
            lat_new = np.dot(self.lat, trans_matrix)
            self.lat = lat_new
        '''


    def rewrite_poscar(self, filename = 'new_poscar'):
        print('please use the new file: ' + filename)
        fout = open(filename, 'w')
        fout.write('strained_poscar\n')
        fout.write('%20.12f\n' % self.zoom)
        for i in range(3):
            fout.write('  %20.12f    %20.12f    %20.12f\n' % (self.lat[i][0], self.lat[i][1], self.lat[i][2]) )
        
        for i in range(len(self.element)):
            fout.write(' '+self.element[i])

        fout.write('\n')
        for i in range(len(self.n_atom)):
            fout.write(' '+str(self.n_atom[i]))
        fout.write('\n')

        fout.write(self.d_or_c+'\n')
        for i in range(self.n_total):
            fout.write('  %20.12f    %20.12f    %20.12f\n' % (self.x_atom[i][0], self.x_atom[i][1], self.x_atom[i][2])) 

def parse_direction(in_arg):
    direction_list = []
    num_direction = len(in_arg) - 2
    for i in range(1, num_direction+1):
        if in_arg[i] in [ 'x', 'X', '1']:
            direction_list.append(1)
        elif in_arg[i] in [ 'y', 'Y', '2']:
            direction_list.append(2)
        elif in_arg[i] in [ 'z', 'Z', '3']:
            direction_list.append(3)
        elif in_arg[i] in [ 'yz', 'zy', 'YZ', 'ZY', '4']:
            direction_list.append(4)
        elif in_arg[i] in [ 'xz', 'zx', 'XZ', 'ZX', '5']:
            direction_list.append(5)
        elif in_arg[i] in [ 'xy', 'yx', 'XY', 'YX', '6']:
            direction_list.append(6)

    return direction_list

    


if __name__ == '__main__':
    #f_sym = open('sym', 'r')
    #sym = f_sym.readline().split()[0]
    #f_sym.close()
    
    if len(sys.argv) >= 3:
        direction_list = parse_direction(sys.argv)
        str_ratio = sys.argv[-1]
        ratio = float(str_ratio)
    else:
        print('need 2 args after script name')
        
    poscar = strain_lattice()
    poscar.strain_all(direction_list, ratio)
    #poscar.rewrite_poscar(filename = 'poscar_'+str(direction)+'_'+str_ratio)
    poscar.rewrite_poscar(filename = 'poscar_'+str_ratio)
