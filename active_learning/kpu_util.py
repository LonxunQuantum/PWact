import os
import glob
import pandas as pd
import numpy as np
import json

from active_learning.util import get_random_nums
from draw_pictures.draw_util import draw_lines
'''
Description: 
param {*} kpu_dir
param {*} method 
    None: kpu_lower= mean(f_kpu) which kpu in range(0.8, 0.95)
    EF: energy and force kpu lowers
Returns: 
Author: WU Xingxing
'''
def get_kpu_lower(kpu_dir, method=None, start_len=None):
    kpu_files = glob.glob(os.path.join(kpu_dir, "*_kpu_info.csv"))
    kpu_info = read_kpu_from_csv(kpu_files)

    force_base, etot_base = None, None
    if method is None:
        res = list(kpu_info["f_kpu"])
        # res = sorted(list(kpu_info["f_kpu"]))
        res = res[-20:]
        # res = res[int(len(res)*0.6):int(len(res)*1)]
        # res = res[int(len(res)*0.5):int(len(res)*0.9)]
        # if start_len is not None:
        #     res = res[start_len:]
        force_base = np.mean(res)
    elif method == "EF":
        res = list(kpu_info["f_kpu"]) #sorted(list(kpu_info["f_kpu"]))
        res = res[int(len(res)*0.6):int(len(res)*1)]
        force_base = np.mean(res)
        res = list(kpu_info["etot_kpu"]) #sorted(list(kpu_info["f_kpu"]))
        res = res[int(len(res)*0.8):int(len(res)*1)]
        etot_base = np.mean(res)
    return force_base, etot_base

"""
@Description :
    read kpu info from kpu_path
    relative_e_f method:
        kpu_res = αi* sqrt(KPU_fi) / (|fi| + L) + α0 * sqrt(KPU_etot) / |etot| (where i = {1, 2, 3}, and sum(α0,...α3)= 1)

    None method (force kpu):
        kpu_res = mean(KPU_fi),i=1,2,3

@Returns     :
    kpu_info: 
    ['batch', 'gpu', 'img_idx', 'natoms', 'etot_lab', 'etot_pre',
        'f_avg_lab', 'f_avg_pre', 'f_x_norm', 'f_y_norm', 'f_z_norm', 'f_kpu',
            'etot_kpu', 'kpu_res']
@Author       :wuxingxing
"""
def read_kpu_from_csv(kpu_dirs):
    kpu_info_list = []
    image_index = 0 
    for file in kpu_dirs:
        kpu_info = pd.read_csv(file, index_col=0, header=0)
        kpu_info.sort_values(by="img_idx", inplace=True, ascending=True)
        kpu_info['img_idx'] = kpu_info['img_idx'] + image_index
        image_index += kpu_info.shape[0]
        kpu_info_list.append(kpu_info)
    
    res = pd.DataFrame(columns=kpu_info.columns)
    for kpu in kpu_info_list:
        res = pd.concat([res, kpu])

    # kpu_info = pd.read_csv(file, index_col=0, header=0) if kpu_info is None else \
    # pd.concat([kpu_info, pd.read_csv(file, index_col=0, header=0)])

    res["kpu_res"] = res["f_kpu"]
    res.sort_values(by="img_idx", inplace=True, ascending=True)
    return res

"""
this function for: kpu = αi* sqrt(KPU_fi) / (|fi| + L) + α0 * sqrt(KPU_etot) / |etot| (where i = {1, 2, 3}, and sum(α0,...α3)= 1)
not be used (2022/12/19)
"""
def cal_relative_kpu(kpu_dir, kpu_name = "valid_kpu_force.csv", kpu_weights=[0.4, 0.2, 0.2, 0.2], constant_l = 0):
    rmse_images_column_name = ["batch","loss", "etot_lab", "etot_pre", "etot_rmse", "kpu_etot","ei_rmse", "f_lab", "f_pre", "f_x_norm", "f_y_norm", "f_z_norm", "f_rmse", "f_kpu", "kpu_x", "kpu_y", "kpu_z"] #kpu_ is force kpu and f_kpu is not used
    avg_images_kpu = pd.DataFrame(columns=rmse_images_column_name)
    rmse_images = pd.read_csv(os.path.join(kpu_dir, kpu_name), index_col=0, header=0) #may nouse
    for i in rmse_images["batch"]:
        i = int(i)
        file_name = "image_{}.csv".format(i)
        force_kpu = pd.read_csv(os.path.join(kpu_dir, file_name), index_col=0, header=0, dtype=float)
        avg_kpu = list(rmse_images.loc[i])
        avg_kpu.extend([force_kpu["kpu_x"].mean(), force_kpu["kpu_y"].mean(), force_kpu["kpu_z"].mean()])
        avg_images_kpu.loc[i] = avg_kpu
    
    # calculate relative kpu of force and energy

    avg_images_kpu["kpu_res"] = kpu_weights[0] * np.sqrt(avg_images_kpu["kpu_etot"]) / (avg_images_kpu["etot_pre"] + constant_l) + \
                                    kpu_weights[1]* np.sqrt(avg_images_kpu["kpu_x"]) / (avg_images_kpu["f_x_norm"] + constant_l) + \
                                        kpu_weights[2]* np.sqrt(avg_images_kpu["kpu_y"]) / (avg_images_kpu["f_y_norm"] + constant_l) + \
                                            kpu_weights[3]* np.sqrt(avg_images_kpu["kpu_z"]) / (avg_images_kpu["f_z_norm"] + constant_l)
    return avg_images_kpu

"""
this function for: kpu = αi* sqrt(KPU_fi) / (|fi| + L) + α0 * sqrt(KPU_etot) / |etot| (where i = {1, 2, 3}, and sum(α0,...α3)= 1)
not be used (2022/12/19)
"""
def cal_force_per_atom(avg_images_kpu):
    avg_images_kpu = avg_images_kpu.reset_index()
    atom_nums = [108, 108, 72, 65]
    sep = int(avg_images_kpu.shape[0]/4)
    avg_images_kpu['f_kpu_atom'] = 0
    for i, v in enumerate(atom_nums):
        avg_images_kpu.loc[i*sep:(i+1)*sep-1]['f_kpu_atom'] = (avg_images_kpu.loc[i*sep:(i+1)*sep-1]['f_kpu']**0.5)/v 
    
    return avg_images_kpu

"""
this function for: kpu = αi* sqrt(KPU_fi) / (|fi| + L) + α0 * sqrt(KPU_etot) / |etot| (where i = {1, 2, 3}, and sum(α0,...α3)= 1)
not be used (2022/12/19)
"""
def cal_energy_per_atom(avg_images_kpu):
    avg_images_kpu = avg_images_kpu.reset_index()
    atom_nums = [108, 108, 72, 65]
    sep = int(avg_images_kpu.shape[0]/4)
    kpu_etot_atom = []
    for i, v in enumerate(atom_nums):
        kpu_etot_atom.extend((avg_images_kpu.loc[i*sep:(i+1)*sep-1]['kpu_etot']**0.5)/v)
        # avg_images_kpu.loc[i*sep:(i+1)*sep-1]['kpu_etot_atom'] = kpu_etot_atom
    
    avg_images_kpu['kpu_etot_atom'] = kpu_etot_atom

    constant_l = 0
    avg_images_kpu["kpu_res_c0"] = 0.4 * np.sqrt(avg_images_kpu["kpu_etot"]) / (avg_images_kpu["etot_pre"] + constant_l) + \
                                    0.2 * np.sqrt(avg_images_kpu["kpu_x"]) / (avg_images_kpu["f_x_norm"] + constant_l) + \
                                        0.2 * np.sqrt(avg_images_kpu["kpu_y"]) / (avg_images_kpu["f_y_norm"] + constant_l) + \
                                            0.2 * np.sqrt(avg_images_kpu["kpu_z"]) / (avg_images_kpu["f_z_norm"] + constant_l)

    constant_l = 0.5
    avg_images_kpu["kpu_res_c05"] = 0.4 * np.sqrt(avg_images_kpu["kpu_etot"]) / (avg_images_kpu["etot_pre"] + constant_l) + \
                                0.2 * np.sqrt(avg_images_kpu["kpu_x"]) / (avg_images_kpu["f_x_norm"] + constant_l) + \
                                    0.2 * np.sqrt(avg_images_kpu["kpu_y"]) / (avg_images_kpu["f_y_norm"] + constant_l) + \
                                        0.2 * np.sqrt(avg_images_kpu["kpu_z"]) / (avg_images_kpu["f_z_norm"] + constant_l)
    constant_l = 0
    avg_images_kpu["kpu_res_c0_f"] = np.sqrt(avg_images_kpu["kpu_x"]) / (avg_images_kpu["f_x_norm"] + constant_l) + \
                                    np.sqrt(avg_images_kpu["kpu_y"]) / (avg_images_kpu["f_y_norm"] + constant_l) + \
                                    np.sqrt(avg_images_kpu["kpu_z"]) / (avg_images_kpu["f_z_norm"] + constant_l)
    
    return avg_images_kpu

"""
@Description :
 select image which need to be labeled in fp step
@Returns     :
@Author       :wuxingxing
"""
def select_image(system_info, work_dir, itername):
    kpu_res_json = json.load(open(os.path.join(system_info["work_root_path"], "kpu_result.json"))) \
        if os.path.exists(os.path.join(system_info["work_root_path"], "kpu_result.json")) else {}

    # get kpu of all training data
    force_kpu_base, etot_base = get_kpu_lower(work_dir.train_kpu_dir)

    kpu_res_json[itername] = {}
    md_kpu_dirs = glob.glob(os.path.join(work_dir.model_dir, "md_*_kpu_dir/*_kpu_info.csv"))
    md_kpu_dirs = sorted(md_kpu_dirs, key=lambda x: int(x.split('/')[-2].split('_')[1]))
    kpu_info = read_kpu_from_csv(md_kpu_dirs)

    #this iter if new phase?
    new_pahse_sign = is_new_phase_md(itername, system_info)
    if new_pahse_sign:
        force_low = force_kpu_base * system_info["kpu_thd"]["kpu_limit_new_phase"][0]
        force_high = force_kpu_base * system_info["kpu_thd"]["kpu_limit_new_phase"][1]
    else:
        force_low = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][0]
        force_high = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][1] 

    max_select = system_info["kpu_thd"]["max_slt"]
    
    kpu_res_json[itername]["force_kpu_lower"] = force_low
    kpu_res_json[itername]["force_kpu_upper"] = force_high

    accuracy = {}
    cadidate = {}
    del_cadidate = {}
    error = {}
    for index, row in kpu_info.iterrows():
        img_idx = int(row['img_idx'])
        if row['f_kpu'] <= force_low:
            accuracy[img_idx] = row['f_kpu']
        elif (row['f_kpu'] > force_low and row["f_kpu"] <= force_high):
            cadidate[img_idx] = row['f_kpu']
        else:
            error[img_idx] = row['f_kpu']

    all_nums = kpu_info.shape[0]
    res_info = "accuracy: {} {};cadidate: {} {};error: {} {}".format(
                    len(accuracy.keys()), round(len(accuracy.keys()) / all_nums, 2), 
                    len(cadidate.keys()), round(len(cadidate.keys()) / all_nums, 2),
                    len(error.keys()), round(len(error.keys()) / all_nums, 2)
    )
    print(itername, ":\n" ,res_info)

    # if nums selected lagger than max select param, randomly remove over images
    if len(cadidate) > max_select:
        mov_list = get_random_nums(0, len(cadidate), int(len(cadidate) - max_select))
        cadi_keys = list(cadidate.keys())
        for index, key in enumerate(cadi_keys):
            if index in mov_list:
                del_cadidate[key] = cadidate.pop(key)

    kpu_select = {}
    kpu_select['res_info'] = res_info
    kpu_select["cadidate"] = list(cadidate.keys())
    kpu_res_json[itername]["kpu_select"] = kpu_select
    json.dump(kpu_res_json, open(os.path.join(system_info["work_root_path"], "kpu_result.json"), "w"), indent=4)

    kpu_select_detail = {}
    kpu_select['accuracy'] = accuracy
    kpu_select['cadidate'] = cadidate
    kpu_select['del_cadidate'] = del_cadidate
    kpu_select['error'] = error
    kpu_select_detail[itername]=kpu_select
    json.dump(kpu_select_detail, open(os.path.join(system_info["work_root_path"], "kpu_select_detail.json"), "w"), indent=4)

def is_new_phase_md(itername, system_info):
    md_index = int(itername.split('.')[-1])
    if "new_phase" in system_info["md_jobs"][md_index].keys():
        return True
    else:
        return False

"""
@Description :
 select image which need to be labeled in fp step
@Returns     :
@Author       :wuxingxing
"""
def select_image_energy_force(system_info, work_dir, itername):
    kpu_res_json = json.load(open(os.path.join(system_info["work_root_path"], "kpu_result.json"))) \
        if os.path.exists(os.path.join(system_info["work_root_path"], "kpu_result.json")) else {}

    #get kpu_lower from training data
    force_kpu_base, etot_kpu_base = get_kpu_lower(work_dir.train_kpu_dir)
    kpu_res_json[itername] = {}

    # get kpu info from mlff md
    # kpu_method1:
    md_kpu_dirs = glob.glob(os.path.join(work_dir.model_dir, "md_*_kpu_dir/*_kpu_info.csv"))
    md_kpu_dirs = sorted(md_kpu_dirs, key=lambda x: int(x.split('/')[-2].split('_')[1]))
    kpu_info = read_kpu_from_csv(md_kpu_dirs)
    # if method =="scal_kpu":
    #     kpu_info, d = scal_kpu(kpu_info, force_kpu_base)
    #     force_low = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][0]
    #     force_high = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][1]
    # else: # None
    force_low = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][0]
    force_high = force_kpu_base * system_info["kpu_thd"]["kpu_limit"][1]
    # etot_low = etot_kpu_base * system_info["kpu_thd"]["kpu_limit"][0]
    # etot_high = etot_kpu_base * system_info["kpu_thd"]["kpu_limit"][1]
    etot_low = etot_kpu_base*1
    etot_high = etot_kpu_base*1.15

    max_select = system_info["kpu_thd"]["max_slt"]
    
    kpu_res_json[itername]["force_kpu_lower"] = force_low
    kpu_res_json[itername]["force_kpu_upper"] = force_high
    kpu_res_json[itername]["etot_kpu_lower"] = etot_low
    kpu_res_json[itername]["etot_kpu_upper"] = etot_high

    accuracy = {}
    cadidate = {}
    del_cadidate = {}
    error = {}
    cad_force = {}
    cad_etot = {}
    for index, row in kpu_info.iterrows():
        img_idx = int(row['img_idx'])
        if row['f_kpu'] <= force_low and row['etot_kpu'] <= etot_low :
            accuracy[img_idx] = [row['f_kpu'],row['etot_kpu']]

        elif (row['f_kpu'] > force_low and row["f_kpu"] <= force_high) or\
                (row['etot_kpu'] > etot_low and row['etot_kpu'] <= etot_high):
            cadidate[img_idx] = [row['f_kpu'],row['etot_kpu']]
            if (row['f_kpu'] > force_low and row["f_kpu"] <= force_high):
                cad_force[img_idx] = row['f_kpu'] 
            else: 
                cad_etot[img_idx] = row['etot_kpu']
        
        elif row['f_kpu'] <= force_low and row['etot_kpu'] > etot_high:
            cadidate[img_idx] = [row['f_kpu'],row['etot_kpu']]
            cad_etot[img_idx] = row['etot_kpu']
        elif row['f_kpu'] > force_high and row['etot_kpu'] <= etot_low:
            cadidate[img_idx] = [row['f_kpu'],row['etot_kpu']]
            cad_force[img_idx] = row['f_kpu'] 

        # elif row["f_kpu"] >= force_high and row['etot_kpu'] >= etot_high:
        #     error[img_idx] = [row['f_kpu'],row['etot_kpu']]
        else:
            error[img_idx] = [row['f_kpu'],row['etot_kpu']]

    comm_cad = []
    cad_f = list(cad_force.keys())
    cad_e = list(cad_etot.keys())
    for i in cad_e:
        if i in cad_f:
            comm_cad.append(i)
    all_nums = kpu_info.shape[0]
    res_info = "accuracy: {} {};cadidate: {} {};cad_etot: {} {};cad_force: {} {};comm: {};error: {} {}".format(
                    len(accuracy.keys()), round(len(accuracy.keys()) / all_nums, 2), 
                    len(cadidate.keys()), round(len(cadidate.keys()) / all_nums, 2),

                    len(cad_etot.keys()), round(len(cad_etot.keys()) / all_nums, 2),
                    len(cad_force.keys()), round(len(cad_force.keys()) / all_nums, 2),
                    len(comm_cad),

                    len(error.keys()), round(len(error.keys()) / all_nums, 2)
    )
    print(itername, ":\n" ,res_info)

    # if nums selected lagger than max select param, randomly remove over images
    if len(cadidate) > max_select:
        mov_list = get_random_nums(0, len(cadidate), int(len(cadidate) - max_select))
        cadi_keys = list(cadidate.keys())
        for index, key in enumerate(cadi_keys):
            if index in mov_list:
                del_cadidate[key] = cadidate.pop(key)

    kpu_select = {}
    kpu_select['res_info'] = res_info
    kpu_select["cadidate"] = list(cadidate.keys())
    kpu_res_json[itername]["kpu_select"] = kpu_select
    json.dump(kpu_res_json, open(os.path.join(system_info["work_root_path"], "kpu_result.json"), "w"), indent=4)

    kpu_select_detail = {}
    kpu_select['accuracy'] = accuracy
    kpu_select['cadidate'] = cadidate
    kpu_select["cad_etot"] = cad_etot
    kpu_select["cad_force"] = cad_force
    kpu_select['del_cadidate'] = del_cadidate
    kpu_select['error'] = error
    kpu_select_detail[itername]=kpu_select
    json.dump(kpu_select_detail, open(os.path.join(system_info["work_root_path"], "kpu_select_detail.json"), "w"), indent=4)
    

'''
Description: 
    scale kpu of md to the same scale 
param {*} kpu_info
param {*} alp0 kpu_lower
Returns: 
    kpu' = a0/a1 * kpu
Author: WU Xingxing
'''
def scal_kpu(kpu_info, alp0):
    alp1 = kpu_info['kpu_res'].min()
    alp2 = kpu_info['kpu_res'].max()
    d = alp1 / alp0
    if d > 1:
       kpu_info['kpu_res'] = kpu_info['kpu_res'] / d
    return kpu_info, d
