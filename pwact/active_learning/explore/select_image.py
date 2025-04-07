from pandas.core.frame import DataFrame
import os

import pandas as pd
import numpy as  np
from pwact.utils.constant import EXPLORE_FILE_STRUCTURE, UNCERTAINTY, SLURM_OUT
from pwact.utils.file_operation import write_to_file, search_files, read_data

from pwact.utils.format_input_output import get_sub_md_sys_template_name
def _select_image(save_dir:str, devi_pd:DataFrame, lower:float, higer:float, max_select:float):
    accurate_pd  = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < lower]
    candidate_pd = devi_pd[(devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] >= lower) & (devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < higer)]
    error_pd     = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] > higer]
    #4. if selected images more than number limitaions, randomly select
    remove_candi = None
    rand_candi = None
    if candidate_pd.shape[0] > max_select:
        rand_candi = candidate_pd.sample(max_select)
        remove_candi = candidate_pd.drop(rand_candi.index)
    
    #5. save select info
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    summary = "Total structures {}    accurate {} rate {:.2f}%    selected {} rate {:.2f}%    error {} rate {:.2f}%\n"\
        .format(devi_pd.shape[0], accurate_pd.shape[0], accurate_pd.shape[0]/devi_pd.shape[0]*100, \
                    candidate_pd.shape[0], candidate_pd.shape[0]/devi_pd.shape[0]*100, \
                        error_pd.shape[0], error_pd.shape[0]/devi_pd.shape[0]*100)

    accurate_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.accurate))
    candi_info = ""
    if rand_candi is not None:
        rand_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate))
        remove_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate_delete))
        candi_info += "Candidate configurations: {}, randomly select {}, delete {}\n        Select details in file {}\n        Delete details in file {}.\n".format(
                candidate_pd.shape[0], rand_candi.shape[0], remove_candi.shape[0],\
                EXPLORE_FILE_STRUCTURE.candidate, EXPLORE_FILE_STRUCTURE.candidate_delete)
    else:
        candidate_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate))
        candi_info += "        Candidate configurations: {}\n    Select details in file {}\n".format(
                candidate_pd.shape[0], EXPLORE_FILE_STRUCTURE.candidate)
            
    error_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.failed))
    
    summary_info = ""

    summary_info += summary
    summary_info += "\nSelect by model deviation force:\n"
    summary_info += "Accurate configurations: {}, details in file {}\n".\
        format(accurate_pd.shape[0], EXPLORE_FILE_STRUCTURE.accurate)
        
    summary_info += candi_info
        
    summary_info += "Error configurations: {}, details in file {}\n".\
        format(error_pd.shape[0], EXPLORE_FILE_STRUCTURE.failed)
    
    write_to_file(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.select_summary), summary_info, "w")
    return summary_info, summary


def sort_model_devi_files(model_devi_files: list):
    md_sys_dict = {}
    for file in model_devi_files:
        md_sys = os.path.basename(os.path.dirname(os.path.dirname(file)))
        sys = int(md_sys.split('.')[3])
        md = int(md_sys.split('.')[1])
        if md not in md_sys_dict.keys():
            md_sys_dict[md] = {sys:[file]}
        else:
            if sys in md_sys_dict[md].keys():
                md_sys_dict[md][sys].append(file)
            else:
                md_sys_dict[md][sys] = [file]
    return md_sys_dict

def select_image(
    md_dir:str, 
    save_dir:str,
    md_job:list,
    devi_name:str,
    lower:float, 
    higer:float
):  
    #1. get model_deviation file
    model_deviation_patten = "{}/{}".format(get_sub_md_sys_template_name(), devi_name)
    model_devi_files = search_files(md_dir, model_deviation_patten)
    model_devi_files = sorted(model_devi_files)
    md_sys_dict = sort_model_devi_files(model_devi_files)
    
    error_pd =None
    accurate_pd =None
    rand_candi =None
    remove_candi =None
    
    error_pd = pd.DataFrame(columns=EXPLORE_FILE_STRUCTURE.devi_columns)
    accurate_pd = pd.DataFrame(columns=EXPLORE_FILE_STRUCTURE.devi_columns)
    rand_candi = pd.DataFrame(columns=EXPLORE_FILE_STRUCTURE.devi_columns)
    remove_candi = pd.DataFrame(columns=EXPLORE_FILE_STRUCTURE.devi_columns)

    for md in md_sys_dict.keys():
        sys_dict = md_sys_dict[md]
        for sys_idx, sys in enumerate(sys_dict.keys()):
            devi_files = sys_dict[sys]
            select_num = md_job[md].select_sys[sys_idx]
            tmp_devi_pd, _base_kpu = read_pd_files(devi_files)
            if len(_base_kpu) > 0: # for kpu upper and lower
                _lower = np.mean(_base_kpu)*lower
                _higer = _lower * higer
            else:
                _lower = lower
                _higer = higer
            tmp_error_pd, tmp_accurate_pd, tmp_rand_candi, tmp_remove_candi = select_pd(tmp_devi_pd, _lower, _higer, select_num)
            error_pd = pd.concat([error_pd, tmp_error_pd])
            accurate_pd = pd.concat([accurate_pd, tmp_accurate_pd])
            rand_candi = pd.concat([rand_candi, tmp_rand_candi])
            remove_candi = pd.concat([remove_candi, tmp_remove_candi])
    right_md, error_md = count_mdstop_info(model_devi_files)
    md_run_info = "A total of {} MD trajectories were run. with {} trajectories correctly executed and {} trajectories normally completed. \nFor detailed information, refer to File {}.".format(len(right_md) + len(error_md), len(right_md), len(error_md), EXPLORE_FILE_STRUCTURE.md_traj_error_record)
    
    summary_info, summary = count_info(save_dir, error_pd, accurate_pd, rand_candi, remove_candi, md_run_info)
    print("Image select result:\n {}\n\n".format(summary_info))

    write_to_file(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.md_traj_error_record), md_run_info, "w")
    details = "\n"
    if len(error_md) > 0:
        details += "\nUnfinished md trajectory directory:\n"
        details += "\n".join(error_md)
    if len(right_md) > 0:
        details += "\n\nCorrectly run md trajectory directory:\n"
        details += "\n".join(right_md)
    write_to_file(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.md_traj_error_record), details, "a")

    return summary

def print_select_image(
    md_dir:str, 
    save_dir:str,
    devi_name:str,
    lower:float, 
    higer:float
):
    #1. get model_deviation file
    model_deviation_patten = "{}/{}".format(get_sub_md_sys_template_name(), devi_name)
    model_devi_files = search_files(md_dir, model_deviation_patten)
    model_devi_files = sorted(model_devi_files)
    md_sys_dict = sort_model_devi_files(model_devi_files)
    
    error_pd =None
    accurate_pd =None
    rand_candi =None
    remove_candi =None

    for md in md_sys_dict.keys():
        sys_dict = md_sys_dict[md]
        for sys_idx, sys in enumerate(sys_dict.keys()):
            devi_files = sys_dict[sys]
            tmp_devi_pd, _base_kpu = read_pd_files(devi_files)
            if len(_base_kpu) > 0: # for kpu upper and lower
                _lower = np.mean(_base_kpu)*lower
                _higer = _lower * higer
            else:
                _lower = lower
                _higer = higer
            tmp_error_pd, tmp_accurate_pd, tmp_rand_candi, tmp_remove_candi = select_pd(tmp_devi_pd, _lower, _higer, 10000000)
            error_pd = pd.concat([error_pd, tmp_error_pd]) if error_pd is not None else tmp_error_pd
            accurate_pd = pd.concat([accurate_pd, tmp_accurate_pd]) if error_pd is not None else tmp_accurate_pd
            rand_candi = pd.concat([rand_candi, tmp_rand_candi]) if error_pd is not None else tmp_rand_candi
            remove_candi = pd.concat([remove_candi, tmp_remove_candi]) if error_pd is not None else tmp_remove_candi
    summary_info, summary = count_info(save_dir, error_pd, accurate_pd, rand_candi, remove_candi)
    
    # summary_info, summary = select_image(save_dir=self.select_dir, 
    #                 devi_pd=devi_pd, 
    #                 lower=self.input_param.strategy.lower_model_deiv_f, 
    #                 higer=self.input_param.strategy.upper_model_deiv_f, 
    #                 max_select=self.input_param.strategy.max_select)
    print("Image select result (lower {} upper {}):\n {}\n\n".format(lower, higer, summary_info))
    return summary


def select_pd(devi_pd:DataFrame, lower:float, higer:float, max_select:float):
    accurate_pd  = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < lower]
    candidate_pd = devi_pd[(devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] >= lower) & (devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] < higer)]
    error_pd     = devi_pd[devi_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] > higer]
    #4. if selected images more than number limitaions, randomly select
    cand_remove_candi = None
    cand_rand_candi = None
    if candidate_pd.shape[0] > max_select:
        cand_rand_candi = candidate_pd.sample(max_select)
        cand_remove_candi = candidate_pd.drop(cand_rand_candi.index)
    else:
        cand_rand_candi = candidate_pd
        cand_remove_candi = pd.DataFrame(columns=cand_rand_candi.columns)
    return error_pd, accurate_pd, cand_rand_candi, cand_remove_candi

def read_pd_files(model_devi_files:list[str]):
    devi_pd = pd.DataFrame()#columns=EXPLORE_FILE_STRUCTURE.devi_columns
    base_force_kpu = []
    if os.path.basename(model_devi_files[0]) == EXPLORE_FILE_STRUCTURE.kpu_model_devi:
        for devi_file in model_devi_files:
                # the data format of devi_file example:
                #     step        etot_kpu      f_kpu_mean       f_kpu_max       f_kpu_min
                #        0            0.00           36.09          107.61            2.00
                #        5            0.00          103.58          327.85            2.00
                # ......
                devi_force = np.loadtxt(devi_file)
                tmp_pd = pd.DataFrame()
                tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] = devi_force[:, 2]#
                tmp_len = int(devi_force.shape[0]*0.01)
                tmp_len = 5 if tmp_len < 5 else tmp_len
                base_force_kpu.extend(devi_force[:, 2][:tmp_len])
                tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[1]] = devi_force[:, 0]
                tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[2]] = os.path.dirname(devi_file)
                devi_pd = pd.concat([devi_pd, tmp_pd])
    else:
        for devi_file in model_devi_files:
            devi_force = read_data(devi_file, skiprows=0)
            tmp_pd = pd.DataFrame()
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[0]] = devi_force[:, 1]
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[1]] = devi_force[:, 0]
            tmp_pd[EXPLORE_FILE_STRUCTURE.devi_columns[2]] = os.path.dirname(devi_file)
            devi_pd = pd.concat([devi_pd, tmp_pd])
    devi_pd.reset_index(drop=True, inplace=True)
    devi_pd["config_index"].astype(int)

    return devi_pd, base_force_kpu


def count_info(save_dir, error_pd, accurate_pd, rand_candi, remove_candi, md_run_info:str=None):
    #5. save select info
    total_num = error_pd.shape[0] + accurate_pd.shape[0] + rand_candi.shape[0] + remove_candi.shape[0]
    cand_num = rand_candi.shape[0] + remove_candi.shape[0]
    summary = "Total structures {}    accurate {} rate {:.2f}%    selected {} rate {:.2f}%    error {} rate {:.2f}%\n"\
        .format(total_num, accurate_pd.shape[0], accurate_pd.shape[0]/total_num*100, \
                    cand_num, cand_num/total_num*100, \
                        error_pd.shape[0], error_pd.shape[0]/total_num*100)
    candi_info = ""
    if remove_candi.shape[0] == 0:
        candi_info += "Candidate configurations: {}\n        Select details in file {}\n".format(
            cand_num, EXPLORE_FILE_STRUCTURE.candidate)
    else:
        candi_info += "Candidate configurations: {}, randomly select {}, delete {}\n        Select details in file {}\n        Delete details in file {}.\n".format(
            cand_num, rand_candi.shape[0], remove_candi.shape[0],\
            EXPLORE_FILE_STRUCTURE.candidate, EXPLORE_FILE_STRUCTURE.candidate_delete)
    summary_info = ""
    summary_info += summary
    summary_info += "\nSelect by model deviation force:\n"
    summary_info += "Accurate configurations: {}, details in file {}\n".\
        format(accurate_pd.shape[0], EXPLORE_FILE_STRUCTURE.accurate)
    summary_info += candi_info
    summary_info += "Error configurations: {}, details in file {}\n\n".\
        format(error_pd.shape[0], EXPLORE_FILE_STRUCTURE.failed)
    if md_run_info is not None:
        summary_info += md_run_info
        summary_info += "\n"

        summary += md_run_info 
        summary += "\n"
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        accurate_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.accurate))
        rand_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate))
        if remove_candi.shape[0] > 0:
            remove_candi.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.candidate_delete))
        error_pd.to_csv(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.failed))
        write_to_file(os.path.join(save_dir, EXPLORE_FILE_STRUCTURE.select_summary), summary_info, "w")
        
    return summary_info, summary


def count_mdstop_info(devi_file_list):
    def read_md_last_line(md_log):
        with open(md_log, "rb") as file:
            file.seek(-2, 2)  # 定位到文件末尾前两个字节
            while file.read(1) != b'\n':  # 逐字节向前查找换行符
                file.seek(-2, 1)  # 向前移动两个字节
            last_line = file.readline().decode().strip()  # 读取最后一行并去除换行符和空白字符
        if "Total wall time" in last_line: #md 正常结束
            return True
        else:
            return False
    # for each md model_deviation file get shape
    # for each md md.log get run time 
    # do compare
    right_list = []
    error_list = []
    for devi_file in devi_file_list:
        devi = np.loadtxt(devi_file)
        end_normal = read_md_last_line(os.path.join(os.path.dirname(devi_file), SLURM_OUT.md_out))
        if end_normal and devi.shape[0] > 1:
            right_list.append(os.path.dirname(devi_file))
        else:
            error_list.append(os.path.dirname(devi_file))
    return right_list, error_list

    