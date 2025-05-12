#!/usr/bin/env python
import os
import signal
import glob
import sys
import json
import argparse
from pwact.utils.constant import TEMP_STRUCTURE, UNCERTAINTY, AL_WORK, AL_STRUCTURE, LABEL_FILE_STRUCTURE, EXPLORE_FILE_STRUCTURE, DFT_STYLE
from pwact.utils.format_input_output import make_iter_name
from pwact.utils.file_operation import write_to_file, del_file_list, search_files, del_dir, copy_dir
from pwact.utils.json_operation import convert_keys_to_lowercase

from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.iter_input import InputParam
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam
from pwact.active_learning.train.train_model import ModelTrian
from pwact.active_learning.train.dp_kpu import ModelKPU
from pwact.active_learning.explore.run_model_md import Explore
from pwact.active_learning.label.labeling import Labeling
from pwact.active_learning.user_input.cmd_infos import cmd_infos

from pwact.active_learning.init_bulk.init_bulk_run import init_bulk_run, scancel_jobs as init_scancel_jobs
from pwact.active_learning.environment import check_envs

from pwact.data_format.configop import extract_pwdata
from pwact.active_learning.explore.select_image import select_image, print_select_image
from pwact.utils.process_tool import kill_process, get_pid
def run_iter():
    system_json = json.load(open(sys.argv[2]))
    if "work_dir" in system_json.keys():
        os.chdir(system_json["work_dir"])
    pid = get_pid()
    with open("./PID", 'w') as wf:
        wf.write(pid)

    system_info = convert_keys_to_lowercase(system_json)
    machine_json = json.load(open(sys.argv[3]))
    machine_info = convert_keys_to_lowercase(machine_json)

    input_param = InputParam(system_info)
    resource = Resource(machine_info, dft_style=input_param.dft_style)
    os.chdir(input_param.root_dir)
    print("The work dir change to {}".format(os.getcwd()))
    record = os.path.abspath(input_param.record_file)
    iter_rec = [0, -1]
    if os.path.isfile(record):
        with open (record) as frec :
            for line in frec :
                if line == '\n':
                    continue
                iter_rec = [int(x) for x in line.split()]
        print ("continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1]))

    ii = -1
    numb_task = 3
    max_tasks = input_param.explore.md_job_num
    while ii < max_tasks:#control by config.json
        ii += 1
        iter_name=make_iter_name(ii)
        for jj in range (numb_task) :
            if ii * max_tasks*10000 + jj <= iter_rec[0] * max_tasks*10000 + iter_rec[1] : 
                continue
            print("current iter is {}".format(iter_name))
            task_name="task %02d"%jj
            print("{} - {}".format(iter_name, task_name))
            if  jj == 0 and ii <= max_tasks: # the last iter, only need to train the model with all datas
                print ("training start: iter {} - task {}".format(ii, jj))
                do_training_work(iter_name, resource, input_param)
                write_to_file(record, "\n{} {}".format(ii, jj), "a") #record_iter
            elif jj == 1 and ii < max_tasks:
                print ("exploring start: iter {} - task {}".format(ii, jj))
                do_exploring_work(iter_name, resource, input_param)
                write_to_file(record, "\n{} {}".format(ii, jj), "a") #record_iter
            elif jj == 2 and ii < max_tasks:
                print ("run_fp: iter {} - task {}".format(ii, jj))
                run_fp(iter_name, resource, input_param)
                write_to_file(record, "\n{} {}".format(ii, jj), "a") #record_iter
            # write_to_file(record, "\n{} {}".format(ii, jj), "a")
            if jj == 2 and not input_param.reserve_work: # delete temp_work_dir under current iteration after the labeling done
                del_file_list([os.path.join(input_param.root_dir, iter_name, TEMP_STRUCTURE.tmp_run_iter_dir)])

    print("Active learning done! \nYou could use cmd 'pwact gather_pwdata' to collect all datas sampled from iterations.")

def run_fp(itername:str, resource : Resource, input_param: InputParam):
    lab = Labeling(itername, resource, input_param)
    #1. if the label work done before, back up and do new work
    lab.back_label()
    #2. make scf work
    if input_param.dft_style == DFT_STYLE.bigmodel:
        lab.make_bigmodel_work()
        lab.do_bigmodel_jobs()
        lab.do_post_bigmodel()
    else:
        lab.make_scf_work()
        #3. do scf work
        lab.do_scf_jobs()
        #4. collect scf configs outcar or movement, then to pwdata format
        lab.do_post_labeling()
    
def do_training_work(itername:str, resource : Resource, input_param: InputParam):
    mtrain = ModelTrian(itername, resource, input_param)
    # 1. if the train work done before, backup the train dir and retrain
    mtrain.back_train()
    # 2. create train work dirs
    mtrain.make_train_work()
    # 3. run training job
    mtrain.do_train_job()
    # 4. do post process after training
    mtrain.post_process_train()
    print("{} done !".format("train_model"))

def do_exploring_work(itername:str, resource : Resource, input_param: InputParam):
    md = Explore(itername, resource, input_param)
    # 1. if the explore work done before, back up explore dir do new explore work
    md.back_explore()
    # 2. make md work files
    md.make_md_work()
    # 3. do md job
    md.do_md_jobs()
    # 4. select images
    if input_param.strategy.uncertainty.upper() == UNCERTAINTY.committee.upper():
        summary = select_image(
                md_dir=md.md_dir, 
                save_dir=md.select_dir,
                md_job=md.md_job,
                devi_name=EXPLORE_FILE_STRUCTURE.get_devi_name(UNCERTAINTY.committee),
                lower=input_param.strategy.lower_model_deiv_f,  
                higer=input_param.strategy.upper_model_deiv_f
        )
        # summary = md.select_image_by_committee()
        # committee: read model deviation file under md file
    elif input_param.strategy.uncertainty.upper() == UNCERTAINTY.kpu.upper():
        summary = uncertainty_analyse_kpu(itername, resource, input_param)
    summary = "{}  {}\n".format(itername, summary)
    write_to_file(os.path.join(input_param.root_dir, EXPLORE_FILE_STRUCTURE.iter_select_file), summary, mode='a')

    if input_param.strategy.direct:
        md.make_drct_work()
        md.do_drct_jobs()
        md.post_drct()
    print("config selection done!")

    # 5. do post process 
    md.post_process_md()
    print("exploring done!")

def uncertainty_analyse_kpu(itername:str, resource : Resource, input_param: InputParam):
    mkpu = ModelKPU(itername, resource, input_param)
    # 1. make kpu work dirs
    mkpu.make_kpu_work()
    # 2. do kpu job
    mkpu.do_kpu_jobs()
    # 3. post process after kpu calculate: select images
    summary = mkpu.post_process_kpu()
    return summary

def init_bulk():
    system_json = json.load(open(sys.argv[2]))
    system_info = convert_keys_to_lowercase(system_json)
    if "work_dir" in system_json.keys():
        os.chdir(system_json["work_dir"])
    pid = get_pid()
    with open("./PID", 'w') as wf:
        wf.write(pid)

    machine_info = convert_keys_to_lowercase(json.load(open(sys.argv[3])))
    input_param = InitBulkParam(system_info)
    resource = Resource(machine_info, job_type=AL_WORK.init_bulk, dft_style=input_param.dft_style, scf_style=input_param.scf_style)

    os.chdir(input_param.root_dir)
    
    print("The work dir change to {}".format(os.getcwd()))
    init_bulk_run(resource, input_param)
    print("Init Bulk Work Done!")

def gather_pwmata(input_cmds):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help="specify the dir above the iterations, the default dir is current dir './'\nthe result could be found in './final_pwdata'", type=str, default='./')
    args = parser.parse_args(input_cmds)
    if not os.path.exists(args.input_dir):
        raise Exception("Error! The input dir {} not exists!".format(args.input_dir))
    pwdata_lists = search_files(args.input_dir, "iter*/{}/{}/*".format(AL_STRUCTURE.labeling,  LABEL_FILE_STRUCTURE.result))
    pwdata_lists = sorted(pwdata_lists)
    save_dir = "./final_pwdata"
    res_data_list = []

    for pwdata in pwdata_lists: # /path/iter.0001/label/result/*
        data_name = os.path.basename(pwdata) 
        iter_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pwdata)))) #iter.0001
        target_dir = os.path.join(save_dir, iter_name, data_name) #./final_pwdata/iter.0001
        copy_dir(pwdata, target_dir)
        for path, dirList, fileList in os.walk(pwdata):
            for _ in fileList:
                if "train.xyz" in _:
                    res_data_list.append(os.path.join(path, _))
        for root, dirs, files in os.walk(pwdata):
                if 'energies.npy' in files:
                    res_data_list.append(root)
        
    result_lines = ["\"{}\",".format(_) for _ in res_data_list]
    result_lines = "\n".join(result_lines)
    # result_lines = result_lines[:-1] # Filter the last ','
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_save_path = os.path.join(save_dir, "final_pwdata_list.txt")
    write_to_file(result_save_path, result_lines, mode='w')
    print("All datas in iterations are:\n")
    print(result_lines)
    print("more details could be found in dir {}\n".format(save_dir))

def print_init_json_template():
    pass

def init_surface():
    pass

'''
description: 
输出主动学习流程的模板json文件
return {*}
author: wuxingxing
'''
def print_run_json_template():
    pass

'''
description: 
    增加一些小工具
        1. 整理出所有的movement到一个目录？
return {*}
author: wuxingxing
'''
def common_tool():
    pass

def environment_check():
    check_envs()

def run_scancel_jobs(iter:int, step:int):
    # train
    itername = make_iter_name(iter)
    if step == 0:
        ModelTrian.kill_job(os.getcwd(), itername)
    # explore
    elif step == 1:
        Explore.kill_job(os.getcwd(), itername)
    # labeling
    elif step ==2:
        Labeling.kill_job(os.getcwd(), itername)

def kill_job():
    # system_json = json.load(open(sys.argv[3]))
    # if "work_dir" in system_json.keys():
    #     os.chdir(system_json["work_dir"])
    try:
        with open("./PID", 'r') as rf:
            pid_str_info = rf.readline().split()
            pid = pid_str_info[1]
            jobid = pid_str_info[3] if "job" in pid_str_info else None
    except :
        raise Exception("Error parsing PID file !")
    kill_process(pid, jobid)
    if sys.argv[2].lower() == "init_bulk":
        # search all jobs
        init_scancel_jobs(os.getcwd())
    elif sys.argv[2].lower() == "run":
        iter_rec = [0, -1]
        if os.path.isfile("./al.record"):
            with open ("./al.record") as frec :
                for line in frec :
                    if line == '\n':
                        continue
                    iter_rec = [int(x) for x in line.split()]
                if iter_rec[1] == 2:
                    iter_rec = [iter_rec[0] + 1, 0]
                else:
                    iter_rec = [iter_rec[0], iter_rec[1] + 1]
            print ("The iter %03d task %02d" % (iter_rec[0], iter_rec[1]))
        else:
            iter_rec = [0, 0]
        run_scancel_jobs(iter_rec[0], iter_rec[1])
    else:
        error_log = "Error parsing command !"
        error_log += "The command of kill supported as 'pwact kill init_bulk' or 'pwact kill run <record.file>'\n"
        raise Exception(error_log)

        # for init_bulk relax or aimd jobs
        
    # for run iters jobs

def filter_test(input_cmds):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--md_dir', help="specify the md dir such as 'iter.0000/temp_run_iter_work/explore/md' or iter.0000/explore/md", type=str, required=True)
    parser.add_argument('-l', '--lower', help="specify lower limit value", type=float, required=True)
    parser.add_argument('-u', '--upper', help="specify upper limit value", type=float, required=True)
    parser.add_argument('-s', '--save', action='store_true', help="if '-s' is set, save the detailed information of the selected configs to CSV files")
    
    args = parser.parse_args(input_cmds)
    if not os.path.exists(args.md_dir):
        raise Exception("ERROR! The input md_dir {} not found!".format(args.md_dir))

    save_dir = os.path.join(os.getcwd(), "filter_test_result") if args.save else None
    summary = print_select_image(
                md_dir=args.md_dir, 
                save_dir=save_dir,
                devi_name=EXPLORE_FILE_STRUCTURE.get_devi_name(UNCERTAINTY.committee),
                lower=args.lower,  
                higer=args.upper
        )

def main():
    environment_check()
    if len(sys.argv) == 1 or "-h".upper() == sys.argv[1].upper() or \
        "help".upper() == sys.argv[1].upper() or "-help".upper() == sys.argv[1].upper() or "--help".upper() == sys.argv[1].upper():
        cmd_infos()

    elif "init_bulk".upper() == sys.argv[1].upper():
        if len(sys.argv) == 2 or "-h".upper() == sys.argv[2].upper() or \
        "help".upper() == sys.argv[2].upper() or "-help".upper() == sys.argv[2].upper() or "--help".upper() == sys.argv[2].upper():
            cmd_infos("init_bulk")
        else:
            init_bulk()
    elif "draw".upper() == sys.argv[1].upper():
        from pwact.active_learning.draw.draw_pictures import draw_pictures
        draw_pictures(sys.argv[2:])

    elif "int_surface".upper() == sys.argv[1].upper():
        init_surface()

    elif "init_json".upper() == sys.argv[1].upper():
        print_init_json_template()

    elif "run_json".upper() == sys.argv[1].upper():
        print_run_json_template()
    
    elif "gather_pwdata".upper() == sys.argv[1].upper():
        gather_pwmata(sys.argv[2:])

    elif "to_pwdata".upper() == sys.argv[1].upper():#these function may use pwdata command
        print("\n\nWarning! This method has been abandoned, new conversion methods refer to the pwdata documentation http://doc.lonxun.com/PWMLFF/Appendix-2/\n\n")
        # to_pwdata(sys.argv[2:])
 
    elif "run".upper() == sys.argv[1].upper():
        if len(sys.argv) == 2 or "-h".upper() == sys.argv[2].upper() or \
        "help".upper() == sys.argv[2].upper() or "-help".upper() == sys.argv[2].upper() or "--help".upper() == sys.argv[2].upper():
            cmd_infos("run")
        else:
            run_iter()
    
    elif "kill".upper() == sys.argv[1].upper():
        if len(sys.argv) == 2 or "-h".upper() == sys.argv[2].upper() or \
        "help".upper() == sys.argv[2].upper() or "-help".upper() == sys.argv[2].upper() or "--help".upper() == sys.argv[2].upper():
            cmd_infos("kill")
        else:
            kill_job()
    
    elif "filter_test".upper() == sys.argv[1].upper() or "filter".upper() == sys.argv[1].upper():
        if len(sys.argv) == 2 or "-h".upper() == sys.argv[2].upper() or \
            "help".upper() == sys.argv[2].upper() or "-help".upper() == sys.argv[2].upper() or "--help".upper() == sys.argv[2].upper():
            cmd_infos("filter")
        else:
            filter_test(sys.argv[2:])

    else:
        print("ERROR! The input cmd {} can not be recognized, please check.".format(sys.argv[1]))
        print("\n\n\nYou can enter the following command.\n\n\n")
        cmd_infos()

if __name__ == "__main__":
    main()
