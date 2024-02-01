import os
import sys
import json
import argparse
from utils.constant import UNCERTAINTY, AL_WORK, PWMAT, LABEL_FILE_STRUCTURE
from utils.format_input_output import make_iter_name
from utils.file_operation import write_to_file, del_file_list, search_files, del_dir, copy_dir
from utils.json_operation import convert_keys_to_lowercase
from utils.gen_format.pwdata import Save_Data

from active_learning.user_input.resource import Resource
from active_learning.user_input.iter_input import InputParam
from active_learning.user_input.init_bulk_input import InitBulkParam
from active_learning.train.train_model import ModelTrian
from active_learning.train.dp_kpu import ModelKPU
from active_learning.explore.run_model_md import Explore
from active_learning.label.labeling import Labeling

from active_learning.init_bulk.init_bulk_run import init_bulk_run
from active_learning.environment import check_envs

def run_iter():
    system_info = convert_keys_to_lowercase(json.load(open(sys.argv[2])))
    machine_info = convert_keys_to_lowercase(json.load(open(sys.argv[3])))
    
    resource = Resource(machine_info)
    input_param = InputParam(system_info)
    os.chdir(input_param.root_dir)
    print("The work dir change to {}".format(os.getcwd()))
    record = input_param.record_file
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
        print("current iter is {}".format(iter_name))
        for jj in range (numb_task) :
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] :
                continue
            task_name="task %02d"%jj
            print("{} - {}".format(iter_name, task_name))
            if   jj == 0:
                print ("training start: iter {} - task {}".format(ii, jj))
                do_training_work(iter_name, resource, input_param)
            elif jj == 1:
                print ("exploring start: iter {} - task {}".format(ii, jj))
                do_exploring_work(iter_name, resource, input_param)
            elif jj == 2:
                print ("run_fp: iter {} - task {}".format(ii, jj))
                run_fp(iter_name, resource, input_param)
            #record_iter
            write_to_file(record, "\n{} {}".format(ii, jj), "a")

def run_fp(itername:str, resource : Resource, input_param: InputParam):
    lab = Labeling(itername, resource, input_param)
    #1. if the label work done before, back up and do new work
    lab.back_label()
    #2. make scf work
    lab.make_scf_work()
    #3. do scf work
    lab.do_scf_jobs()
    #4. collect scf configs outcar or movement
    lab.collect_scf_configs()
    #5. change the movement format to pwdata format
    aimd_list = lab.get_aimd_list()
    if len(aimd_list) > 0:
        for aimd_file in aimd_list:
            save_name = os.path.basename(os.path.dirname(aimd_file))
            Save_Data(data_path=aimd_file, 
                datasets_path=lab.result_dir, 
                save_name=save_name,
                train_ratio = input_param.train.train_valid_ratio, 
                random = input_param.train.data_shuffle, 
                format=resource.dft_style)
    #6. collect the files of this iteration to label/result dir
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
        summary = md.select_image_by_committee()
        # committee: read model deviation file under md file
    elif input_param.strategy.uncertainty.upper() == UNCERTAINTY.kpu.upper():
        summary = uncertainty_analyse_kpu(itername, resource, input_param)
    print(summary)
    print("config selection done!")
    # 5. do post process after lammps md running
    md.post_process_md()
    print("lammps md done!")
    return summary

def uncertainty_analyse_kpu(itername:str, resource : Resource, input_param: InputParam):
    mkpu = ModelKPU(itername, resource, input_param)
    # 1. make kpu work dirs
    mkpu.make_kpu_work()
    # 2. do kpu job
    mkpu.do_kpu_jobs()
    # 3. post process after kpu calculate: select images
    mkpu.post_process_kpu()

def init_bulk():
    system_info = convert_keys_to_lowercase(json.load(open(sys.argv[2])))
    machine_info = convert_keys_to_lowercase(json.load(open(sys.argv[3])))
    input_param = InitBulkParam(system_info)
    resource = Resource(machine_info, job_type=AL_WORK.init_bulk)

    os.chdir(input_param.root_dir)
    
    print("The work dir change to {}".format(os.getcwd()))
    init_bulk_run(resource, input_param)
    print("Init Bulk Work Done!")

def to_pwdata(input_cmds:list):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input outcars or movement files', nargs='+', type=str, default=None)
    parser.add_argument('-f', '--format', help="specify input file format, 'vasp' or pwmat", type=str, default="pwmat")
    parser.add_argument('-s', '--savepath', help='specify stored directory', type=str, default='PWdata')
    parser.add_argument('-o', '--train_valid_ratio', help='specify stored directory', type=float, default=0.8)
    parser.add_argument('-r', '--data_shuffle', help='specify stored directory', type=bool, default=True)
    parser.add_argument('-w', '--work_dir', help='specify work dir', type=str, default='./')
    
    args = parser.parse_args(input_cmds)
    os.chdir(args.work_dir)
    for config in args.input:
        Save_Data(data_path=config, 
        datasets_path=args.savepath,
        train_ratio = args.train_valid_ratio, 
        random = args.data_shuffle, 
        format= args.format)

def extract_pwmata(input_cmds):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='specify input dir you want to extract pwdatas', type=str, default=None)
    parser.add_argument('-s', '--save_dir', help="specify the dir to save the extract result", type=str, default="extract_result")
    args = parser.parse_args(input_cmds)
    if not os.path.exists(args.input):
        raise Exception("Error! The input dir {} not exists!".format(args.input))
    pwdata_lists = search_files(args.input, "iter*/{}".format(LABEL_FILE_STRUCTURE.result))
    if os.path.exists(args.save_dir):
       del_dir(args.save_dir)
    os.makedirs(args.save_dir)
    for pwdata in pwdata_lists:
        iter_name = os.path.basename(os.path.dirname(pwdata))
        target_dir = os.path.join(args.save_dir, iter_name)
        copy_dir(pwdata, target_dir)

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
    输出大的命令选项
return {*}
author: wuxingxing
'''
def print_cmd():
    from active_learning.user_input.cmd_infos import cmd_infos
    print(cmd_infos())

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

def main():
    environment_check()
    if len(sys.argv) == 1 or "-h".upper() in sys.argv[1].upper() or "help".upper() in sys.argv[1].upper():
        print_cmd()

    elif "init_bulk".upper() in sys.argv[1].upper():
        init_bulk()

    elif "int_surface".upper() in sys.argv[1].upper():
        init_surface()

    elif "run".upper() in sys.argv[1].upper():
        run_iter()

    elif "init_json".upper() in sys.argv[1].upper():
        print_init_json_template()

    elif "run_json".upper() in sys.argv[1].upper():
        print_run_json_template()
    
    elif "pwdata".upper() in sys.argv[1].upper():
        to_pwdata(sys.argv[2:])
    
    elif "extract_pwdata".upper() in sys.argv[1].upper():
        extract_pwmata(sys.argv[2:])


if __name__ == "__main__":
    main()
