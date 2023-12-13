from draw_pictures.draw_util import draw_lines
from active_learning.kpu_util import read_kpu_from_csv, get_kpu_lower
import json
import os

def read_div_f():
    iter = "iter.0001"
    iter_dpgen = "iter.000001"
    gap = 10
    dt = 2
    div_path = "/share/home/wuxingxing/datas/dpgen_al/Cu_bulk/{}/01.model_devi/task.000.000000/model_devi.out".format(iter_dpgen)
    with open(div_path, 'r') as rf:
        lines = rf.readlines()
    head = lines[0]
    res = {}
    for i, v in enumerate(lines[1:]):
        res[i*gap] = float(v.split()[-1])

    picture_path = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/{}/{}-avg_devi_f.png".format(iter,iter)
    draw_lines([list(res.keys())], [list(res.values())], picture_path,\
        [''], "dpgen force deviation step {} fs".format(dt), "force deviation", "fs")
    
"""
@Description :
 print the kpu info or save 
@Returns     :
@Author       :wuxingxing
"""
def print_kpu_info():
    iter = "iter.0001"
    gap = 10
    dt = 2
    kpu_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/{}/training/model_dir/train_kpu_dir".format(iter)
    md_kpu_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/{}/training/model_dir/md_kpu_dir".format(iter)
    save_dir = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/{}".format(iter)
    kpu_res = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/kpu_result.json"

    lower = get_kpu_lower(kpu_dir)
    i0_md_kpu = read_kpu_from_csv(md_kpu_dir)
    # i0_md_kpu = i0_md_kpu.loc[i0_md_kpu['img_idx'].isin([_ for _ in range(0,200,5)])]
     # read kpu res
    kpu_res = json.load(open(kpu_res))
    kpu_upper = kpu_res[iter]['kpu_upper']
    kpu_lower = kpu_res[iter]['kpu_lower']
    draw_lines( [i0_md_kpu['img_idx']*gap], [i0_md_kpu['f_kpu']], os.path.join(save_dir, "{}_md_kpu.png".format(iter)),\
        [''], "force kpu of dpkf md, step {} fs, base: {}".format(dt,  round(lower,2)), "force kpu", "fs")
    # print(i0_tra_kpu[['img_idx', 'etot_lab', 'etot_pre']].loc[i0_tra_kpu['img_idx'].isin([_ for _ in range(0,200,5)])])

    # draw_lines( [i0_md_kpu['img_idx']*4], [i0_md_kpu['f_kpu']], os.path.join(save_dir, "dpkf_md_force_kpu_800.png"),\
    #     [''], "force kpu of dpkf md, step {} fs".format(1), "force kpu", "fs")
    
    # draw_lines( [i0_md_kpu['img_idx']*4], [i0_md_kpu['etot_kpu']], os.path.join(save_dir, "dpkf_md_etot_kpu_800.png"),\
    #     [''], "etot kpu of dpkf md, step {} fs".format(1), "etot kpu", "fs")

    # draw_lines( [i0_md_kpu['img_idx']*4], [abs(i0_md_kpu['fp_etot'] - i0_md_kpu['etot_pre'])], os.path.join(save_dir, "dpkf_md_kpu_1000.png"),\
    #     [''], "force kpu of dpkf md, step {} fs".format(1), "force kpu", "fs")

'''
Description: 
Returns: 
Author: WU Xingxing
'''
def get_undone_task():
    dir = "/share/home/wuxingxing/datas/dpgen_al/Cu_bulk/work_dir/scf/6b1df61044b216b4cda5f0b3b696fc313603c65c"
    import glob
    dir_list = glob.glob(os.path.join(dir, "task.000.*"))
    res = {}
    res["done"] = []
    res["error"] = []
    for d in dir_list:
        outfile = os.path.join(dir, d, "output")
        if os.path.exists(outfile):
            with open(outfile, 'r') as rf:
                line = rf.readlines()[-1]
            if "total computation time" in line:
                res["done"].append(d)
            else:
                res["error"].append(d)
        else:
            print("{} file not exist".format(outfile))
    print(res)

def get_cadidate():
    dir = ""
if __name__ == "__main__":
    # read_div_f()
    # print_kpu_info()
    # get_undone_task()
    get_cadidate()