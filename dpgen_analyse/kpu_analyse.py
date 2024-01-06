import json
import numpy as np
import pandas as pd
import os
import glob
from dpgen_analyse.kpu_util import read_kpu_from_csv, get_kpu_lower

"""
@Description :
 calculate force rmse, data from kpu_dir in which has force info
@Returns     :
@Author       :wuxingxing
"""
def calculate_rmse(kpu_dir):
    kpu_info = read_kpu_from_csv(kpu_dir)
    img_force_file_lists = get_kpu_file_lists(kpu_dir, "img")
    img_force_file_lists = sorted(img_force_file_lists, key=lambda x: int(x.split('_')[3]))
    force_info = pd.DataFrame(columns=['img_idx', 'f_rmse'])
    for img in img_force_file_lists:
        img_path = os.path.join(kpu_dir, img)
        #['atom_index', 'kpu_x', 'kpu_y', 'kpu_z', 'f_x', 'f_y', 'f_z', 'f_x_pre', 'f_y_pre', 'f_z_pre']
        img_df = pd.read_csv(img_path, index_col=0, header=0, dtype=float)
        y_lab = img_df[["f_x", "f_y", "f_z"]].to_numpy()
        y_pre = img_df[["f_x_pre", "f_y_pre", "f_z_pre"]].to_numpy()
        rmse = np.linalg.norm(y_lab-y_pre, ord=2)/len(y_lab)**0.5
        force_info.loc[force_info.shape[0]] = [int(img.split("_")[3]), rmse]
    force_info['f_kpu'] = kpu_info['f_kpu'].to_numpy()
    
    etot_rmse = []
    for index, row in kpu_info.iterrows():
        y_lab = np.array([row['etot_lab']])
        y_pre = np.array([row['etot_pre']])
        rmse = np.linalg.norm(y_lab-y_pre, ord=2)/len(y_lab)**0.5
        etot_rmse.append(rmse)
    force_info['etot_kpu'] = kpu_info['etot_kpu'].to_numpy()
    force_info['etot_rmse'] = etot_rmse
    return force_info

"""
@Description :
 select images by etot kpu
 result: same as force kpu
@Returns     :
@Author       :wuxingxing
"""
def select_by_etot_kpu():
    kpu_dir = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cu_slab_1500k_system/iter.0000/training/kpu_dir"
    kpu_info = read_kpu_from_csv(kpu_dir)
    res = sorted(list(kpu_info["etot_kpu"]))
    res = res[int(len(res)*0.8):int(len(res)*0.95)]
    kpu_dir = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cu_slab_1500k_system/iter.0000/training/md_kpu_dir"
    kpu_info_md = read_kpu_from_csv(kpu_dir)
    low = np.mean(res)
    high = low * 5
    accuracy = {}
    cadidate = {}
    error = {}
    for index, row in kpu_info_md.iterrows():
        img_idx = int(row['img_idx'])
        if row['kpu_res'] <= low:
            accuracy[img_idx] = row['etot_kpu']
        elif row["kpu_res"] >= high:
            error[img_idx] = row['etot_kpu']
        else:
            cadidate[img_idx] = row['etot_kpu']
    print(accuracy)
    print(cadidate)
    print(error)
    print()

"""
@Description :
 print the kpu info or save 
@Returns     :
@Author       :wuxingxing
"""
def print_kpu_info():
    kpu_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/training/model_dir/train_kpu_dir"
    save_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000"
    i0_tra_kpu = read_kpu_from_csv(kpu_dir)
    lower = get_kpu_lower(kpu_dir)
    print(lower)

    kpu_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/training/model_dir/md_kpu_dir"
    i0_md_kpu = read_kpu_from_csv(kpu_dir)
    i0_md_kpu = i0_md_kpu.loc[i0_md_kpu['img_idx'].isin([_ for _ in range(0,200,5)])]
    # print(i0_tra_kpu[['img_idx', 'etot_lab', 'etot_pre']].loc[i0_tra_kpu['img_idx'].isin([_ for _ in range(0,200,5)])])

    res = pd.DataFrame()
    # draw_lines( [i0_md_kpu['img_idx']*4], [i0_md_kpu['f_kpu']], os.path.join(save_dir, "dpkf_md_force_kpu_800.png"),\
    #     [''], "force kpu of dpkf md, step {} fs".format(1), "force kpu", "fs")
    
    # draw_lines( [i0_md_kpu['img_idx']*4], [i0_md_kpu['etot_kpu']], os.path.join(save_dir, "dpkf_md_etot_kpu_800.png"),\
    #     [''], "etot kpu of dpkf md, step {} fs".format(1), "etot kpu", "fs")

    # draw_lines( [i0_md_kpu['img_idx']*4], [abs(i0_md_kpu['fp_etot'] - i0_md_kpu['etot_pre'])], os.path.join(save_dir, "dpkf_md_kpu_1000.png"),\
    #     [''], "force kpu of dpkf md, step {} fs".format(1), "force kpu", "fs")
    print(res)

    # read kpu res
    kpu_res = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/kpu_result.json"
    kpu_res = json.load(open(kpu_res))
    i0 = list(kpu_res['iter.0000']['kpu_select']['cadidate'].keys())
    i0 = [int(_)*4 for _ in i0]
    print(i0)

def kpu_random_select():
    root_dir = "/share/home/wuxingxing/codespace/active_learning_mlff_multi_job/al_dir/cu_bulk_system/iter.0005/training/model_dir"
    kpu_files = glob.glob(os.path.join(root_dir,"md_*_kpu_dir", "*_kpu_info.csv"))
    kpu_files = sorted(kpu_files, key=lambda x: int(x.split('/')[-2].split('_')[1]))
    kpu_info = read_kpu_from_csv(kpu_files)

    
if __name__ == "__main__":
    # save_path = "/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/multi_gpu_cu_slab_1500k_system/log_dir"
    # kpu_dir = "/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/multi_gpu_cu_slab_1500k_system/init_data_train_kpu_dir"
    # train_rmse = calculate_rmse(kpu_dir)
    # train_rmse.to_csv(os.path.join(save_path, "init_data_train_rmse_50.csv"))
    # kpu_dir = "/home/wuxingxing/datas/system_config/cu_4phases_system/init_models/multi_gpu_cu_slab_1500k_system/init_data_val_kpu_dir"
    # valid_rmse = calculate_rmse(kpu_dir)
    # valid_rmse.to_csv(os.path.join(save_path, "init_data_valid_rmse_50.csv"))
    # count_etot_kpu()
    # print_kpu_info()
    # calculate_rmse()
    kpu_random_select()