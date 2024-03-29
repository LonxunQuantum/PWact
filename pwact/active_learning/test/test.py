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
    
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--config', help="specify config file path of config", type=str, default='atom.config')
    # parser.add_argument('-f', '--format', help="specify config file format config", type=str, default='pwmat/config')
    # parser.add_argument('-k', '--kspacing', help="specify the kspacing, the default 0.5", type=float, default=0.5)
    # args = parser.parse_args()
    # make_kspacing_kpoints(config=args.config, format=args.format, kspacing=args.kspacing)
    make_kspacing_kpoints(config="/data/home/wuxingxing/datas/al_dir/HfO2/dftb/init_bulk_hfo2/temp_init_bulk_work/scf/init_config_0/init/0_aimd/0-scf/POSCAR",
     format="vasp/poscar", kspacing=0.5)

    # work_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory"
    # data_list = [
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_000_2650",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_001_2650",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_002_2650",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_003_2650",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_004_3858",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_005_3860",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_006_3860",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_007_3859",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_008_700",
    #     "/data/home/wuxingxing/datas/PWMLFF_library_data/HfO2/hfo2_dpgen/HfO2_liutheory/mvms/sys_009_700"]
    # work_dir = "/data/home/wuxingxing/datas/al_dir/HfO2/baseline_model"

    # train_job = os.path.join(work_dir, "init_model/train.job")
    # train_json = os.path.join(work_dir, "init_model/train.json")

    # json_dict = json.load(open(train_json))
    # for data in data_list:
    #     data_dir = os.path.join(work_dir, os.path.basename(data))
    #     if os.path.exists(data_dir):
    #         shutil.rmtree(data_dir)
    #     os.makedirs(data_dir)
    #     shutil.copyfile(train_job, os.path.join(data_dir, "train.job"))
    #     json_dict["datasets_path"].append(data)
    #     json.dump(json_dict, open(os.path.join(data_dir, "train.json"), "w"), indent=4)
    
