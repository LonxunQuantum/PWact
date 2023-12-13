import os, sys, glob
import numpy as np
import pandas as pd
from active_learning.kpu_util import read_kpu_from_csv, get_kpu_lower
from active_learning.util import get_random_nums

low_base = 2.5
high_base=5
max_select = 200

def select(iter, low_base, high_base):
    itername = os.path.basename(iter)
    kpu_dir_i0 = os.path.join(os.path.dirname(iter), 'iter.0000', "training/model_dir/train_0_kpu_dir")
    kpu_files_i0 = glob.glob(os.path.join(kpu_dir_i0, "*_kpu_info.csv"))
    kpu_info_i0 = read_kpu_from_csv(kpu_files_i0)
    
    kpu_dir = os.path.join(iter, "training/model_dir/train_0_kpu_dir")
    force_base, etot_base = get_kpu_lower(kpu_dir, start_len = None)

    md_kpu_dirs = glob.glob(os.path.join(iter, "training/model_dir", "md_*_kpu_dir/*_kpu_info.csv"))
    md_kpu_dirs = sorted(md_kpu_dirs, key=lambda x: int(x.split('/')[-2].split('_')[1]))
    kpu_info = read_kpu_from_csv(md_kpu_dirs)
    
    force_low = force_base * low_base
    force_high = force_base * high_base

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

    # if nums selected lagger than max select param, randomly remove over images
    if len(cadidate) > max_select:
        mov_list = get_random_nums(0, len(cadidate), int(len(cadidate) - max_select))
        cadi_keys = list(cadidate.keys())
        for index, key in enumerate(cadi_keys):
            if index in mov_list:
                del_cadidate[key] = cadidate.pop(key)

    kpu_select = {}
    kpu_select['res_info'] = res_info
    kpu_select['accuracy'] = accuracy
    kpu_select['cadidate'] = cadidate
    kpu_select['del_cadidate'] = del_cadidate
    kpu_select['error'] = error

    return len(accuracy), len(cadidate), len(del_cadidate), len(error), force_low, force_high, res_info, kpu_select

def main():
    dir = "/data/home/wuxingxing/al_dir/ni"
    iters = glob.glob(os.path.join(dir, "iter.*"))
    iters = sorted(iters, key=lambda x: int(x.split('/')[-1].split('.')[-1]))
    accuracy = []
    cadidate = []
    error = []
    for i, iter in enumerate(iters):
        if i % 4 == 0:
            low_base = 1.75
            high_base=4
        else:
            low_base = 1
            high_base=2
        acc, cad, del_cad, err, lower, higher,res_info, kpu_select = select(iter,low_base,high_base)
        print(os.path.basename(iter), " : " ,res_info)
        accuracy.append(acc)
        cadidate.append(cad+del_cad)
        error.append(err)
    print("acc: {}, cad:{}, err:{}".format(sum(accuracy), sum(cadidate), sum(error)))
    print()

def cat_kpu_info():
    iter='0004'
    dir = "/data/home/wuxingxing/al_dir/ni/iter.{}".format(iter)
    save_path = "/data/home/wuxingxing/al_dir/ni/{}_md.csv".format(iter)
    md_kpu_dirs = glob.glob(os.path.join(dir, "training/model_dir", "md_*_kpu_dir/*_kpu_info.csv"))
    md_kpu_dirs = sorted(md_kpu_dirs, key=lambda x: int(x.split('/')[-2].split('_')[1]))
    kpu_info = read_kpu_from_csv(md_kpu_dirs)
    kpu_info.to_csv(save_path)

if __name__=="__main__":
    # main()
    cat_kpu_info()