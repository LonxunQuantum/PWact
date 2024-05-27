import numpy as np
import argparse
import os
import shutil
import glob
import json
from pwdata.main import Config
def make_kspacing_kpoints(config, format, kspacing):
    config = Config(format=format, data_path=config)
    # with open(config, "r") as fp:
    #     lines = fp.read().split("\n")
    # box = []
    # for idx, ii in enumerate(lines):
    #     if "LATTICE" in ii.upper():
    #         for kk in range(idx + 1, idx + 1 + 3):
    #             vector = [float(jj) for jj in lines[kk].split()[0:3]]
    #             box.append(vector)
    #         box = np.array(box)
    #         rbox = _reciprocal_box(box)
    box = config.images.lattice
    rbox = _reciprocal_box(box)
    kpoints = [
        max(1, round(2 * np.pi * np.linalg.norm(ii) / kspacing)) for ii in rbox
    ]
    #DP pwmat multi kpoints will be slow, use round not ceil
    # kpoints = [
    #     max(1, (np.ceil(2 * np.pi * np.linalg.norm(ii) / ks).astype(int)))
    #     for ii, ks in zip(rbox, [kspacing,kspacing,kspacing])
    # ]
    ret = ""
    ret += "%d %d %d 0 0 0 " % (kpoints[0], kpoints[1], kpoints[2])
    print(ret)
    return ret
    
def _reciprocal_box(box):
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    return rbox

def get_energy_dftb_vasp():
    aimd_dir = "/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/temp_init_bulk_work/aimd"#/init_config_0/init/0_aimd
    scf_dir = "/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/temp_init_bulk_work/scf"#init_config_0/init/0_aimd/0-scf
    save_file = "/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/energy_count_xtb.txt"
    aimd_dir = glob.glob(os.path.join(aimd_dir, "init_config_*"))
    aimd_dir = sorted(aimd_dir, key=lambda x: int(os.path.basename(x).split('_')[-1]))

    save_text = []
    for aimd in aimd_dir:
        mvm_config = Config(format="pwmat/movement", data_path=os.path.join(aimd, "init/0_aimd/MOVEMENT"))
        scf_list = glob.glob(os.path.join(scf_dir, os.path.basename(aimd), "init/0_aimd/*-scf"))
        scf_list = sorted(scf_list, key=lambda x: int(os.path.basename(x).split('-')[0]))
        for scf in scf_list:
            index = int(os.path.basename(scf).split('-')[0])
            scf_config = Config(format="vasp/outcar", data_path=os.path.join(scf, "OUTCAR"))
            if index == 0:
                base = scf_config.images[0].Ep - mvm_config.images[index].Ep
            save_text.append("aimd {} index {} dftb_energy {} vasp_energy {} vasp_just {}"\
                .format(os.path.basename(aimd), index, \
                    mvm_config.images[index].Ep, scf_config.images[0].Ep,\
                        scf_config.images[0].Ep - base))
    
    with open(save_file, 'w') as wf:
        for line in save_text:
            wf.write(line)
            wf.write("\n")
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--config', help="specify config file path of config", type=str, default='atom.config')
    # parser.add_argument('-f', '--format', help="specify config file format config", type=str, default='pwmat/config')
    # parser.add_argument('-k', '--kspacing', help="specify the kspacing, the default 0.5", type=float, default=0.5)
    # args = parser.parse_args()
    # make_kspacing_kpoints(config=args.config, format=args.format, kspacing=args.kspacing)
    # make_kspacing_kpoints(config="/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/temp_init_bulk_work/scf/init_config_0/init/0_aimd/0-scf/POSCAR",
    #  format="vasp/poscar", kspacing=0.5)
    get_energy_dftb_vasp()
