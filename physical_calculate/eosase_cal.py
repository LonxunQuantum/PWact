import os, sys, glob, argparse
import pandas as pd
import numpy as np
import ase
import ase.eos
import ase.units

def read_ep():
    pre = pd.read_csv(csv_path, index_col=0, header=0, dtype=float)
    pre.sort_values(by="img_idx", inplace=True, ascending=True)
    print("etot_pre:")
    print(list(pre['etot_pre']))
    print("etot_lab:")
    print(list(pre['etot_lab']))
    return pre['etot_pre']

def read_volumn():
    a = np.loadtxt(volumn_path)
    print("volumns:")
    print(a)
    return a

def eosase():
    volumes = read_volumn()
    energies = read_ep()
    EOS = ase.eos.EquationOfState(volumes, energies)
    v0, e0, B = EOS.fit()
    print('v0, a0_if_cubic: ', v0, v0 ** (1.0/3.0))
    print('B: %E GPa' % (B / ase.units.kJ * 1.0e24))  # 1kJ=6.241509125883258 E+21 eV
    print('e0: ', e0)
    EOS.plot(save_path)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--csv_path', help='specify the prediction csv file', type=str, default='/share/home/wuxingxing/al_dir/cu_system/iter.0029/training/model_dir_adam_bignet_1000_epoch/valid_adam_bignet/prediction.csv')
    parser.add_argument('-v', '--volumn_path', help='specify the prediction volumn_path', type=str, default='/share/home/wuxingxing/al_dir/cu_system/physical_character/v0_longer/volumn.txt')
    parser.add_argument('-s', '--save_path', help='specify the prediction csv file', type=str, default='/share/home/wuxingxing/al_dir/cu_system/iter.0029/training/model_dir_adam_bignet_1000_epoch/valid_adam_bignet/bm.png')
    args = parser.parse_args()

    csv_path = args.csv_path
    volumn_path = args.volumn_path
    save_path = args.save_path
    eosase()
    

