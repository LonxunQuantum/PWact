import dpdata
import glob, os
import argparse

'''
Description: convert lammpstrajs in traj_dir to MOVEMENT
param {*} traj_dir
param {*} type_map
Returns: 
Author: WU Xingxing
'''
def traj2movement(traj_dir, movement_save_path, type_map = ["Cu"]):
    trajs = glob.glob(os.path.join(traj_dir, "*lammpstrj"))
    trajs = sorted(trajs, key= lambda x: int(x.split('/')[-1].split('.')[0]))
    atom_list = []
    for traj in trajs:
        index = traj.split('/')[-1].split('.')[0]
        a = dpdata.System(traj, fmt='lammps/dump',type_map=type_map)
        atom_save_path = '{}/atom_{}.config'.format(traj_dir, index)
        a.to('pwmat/atom.config', file_name=atom_save_path)
        atom_list.append(atom_save_path)
    combine_files(atom_list, movement_save_path)
    cmd = "rm {}/atom_*.config".format(traj_dir)
    res = os.system(cmd)
    assert(res == 0)
    print("trajs to movement done.")

def combine_files(atom_list, target_file):
    with open(target_file, 'w') as outfile:
        for file in atom_list:     
            with open(file, 'r') as infile:     
                outfile.write(infile.read()) 
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

#dpdadpdata.LabeledSystem('MOVEMENT_more_calc6', fmt='MOVEMENT').to('deepmd/raw', 'test')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='specify directory of lammpstrjs', type=str, default='.')
    parser.add_argument('-t', '--type-map', help='specify type_map as:  -t Cu C ', nargs="*", default=["Cu"])
    parser.add_argument('-s', '--save-path', help='specify movement save path', type=str, default='./MOVEMENT')
    args = parser.parse_args()
    print(args.directory, args.save_path, args.type_map)
    traj2movement(args.directory, args.save_path, args.type_map)
    # traj2movement()