import numpy as np
import pandas as pd
import argparse
import glob
import os

def collect_model_deivs(dir):
    res_all = pd.DataFrame(columns=['iter','index', 'max_devi_f', 'min_devi_f', 'avg_devi_f'])
    res_info = pd.DataFrame(columns=['iter', 'mean_max_devi_f', 'mean_min_devi_f', 'mean_avg_devi_f'])
    
    model_deiv_paths = glob.glob(os.path.join(dir, "iter.*", "01.model_devi/task.000.000000/model_devi.out"))
    
    for devif_path in model_deiv_paths:
        with open(devif_path, "r") as rf:
            lines = rf.readlines()
        min = []
        max = []
        avg = []
        iters = []
        indexs = []
        iter = devif_path.split('/')[7].split('.')[1]
        res_pd = pd.DataFrame(columns=['iter','index', 'max_devi_f', 'min_devi_f', 'avg_devi_f'])

        for i in lines[2:]:
            i = i.strip()
            step, max_devi_v, min_devi_v, avg_devi_v, \
                max_devi_f, min_devi_f, avg_devi_f, = \
                [float(_.strip()) for _ in i.split()]
            max.append(max_devi_f)
            min.append(min_devi_f)
            avg.append(avg_devi_f)
            iters.append(iter)
            indexs.append(i)

        res_pd['iter'] = iters
        res_pd['index'] = indexs
        res_pd['max_devi_f'] = max
        res_pd['min_devi_f'] = min
        res_pd['avg_devi_f'] = avg
        res_info.loc[res_info.shape[0]] = \
            [iter, res_pd['max_devi_f'].mean(), res_pd['min_devi_f'].mean(),res_pd['avg_devi_f'].mean()]
        res_pd.to_csv(os.path.join(dir, "{}.csv".format(iter)))
        res_all = pd.concat([res_all,res_pd])
    
    res_all.to_csv(os.path.join(dir, "all_model_deiv_details.csv"))
    print(res_info)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--devif_path', help='specify input force file', type=str, default='model_devi_600k_dft.out')

    args = parser.parse_args()

    devif_path = args.devif_path

    collect_model_deivs(devif_path)