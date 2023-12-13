import dpdata
import os
import glob
import argparse

def convert_lammpstrjs_to_movement(source_dir, save_path, type_map):
    trjfiles = glob.glob(os.path.join(source_dir, "*lammpstrj"))
    atom_configs = []
    for i in trjfiles():
        a = dpdata.System('*trj',fmt='lammps/dump',type_map=type_map)
        a.to('pwmat/atom.config',filename)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-dir', help='specify lammpstrj dir', type=str, default='.')
    parser.add_argument('-m', '--save-path', help='specify movement save path', type=str, default='./MOVEMENT')
    parser.add_argument('-t', '--convert-type', help='specify data convert type, -t: trj2mov ', type=str, default='')
    parser.add_argument('-a', '--type-map', help='specify atom typse [29, 8] ', type=list)
    
    args = parser.parse_args()

    source_dir = args.source_dir
    save_path = args.save_path
    convert_type = args.convert_type
    type_list = args.type_map
    if convert_type == "trj2mov":
        convert_lammpstrjs_to_movement(source_dir, save_path, type_list)
