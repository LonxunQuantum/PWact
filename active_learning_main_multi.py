import os
import sys
import json
# import argparse
from utils.constant import UNCERTAINTY, AL_WORK
from utils.format_input_output import make_iter_name
from utils.file_operation import write_to_file

from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam
from active_learning.user_input.init_bulk_input import InitBulkParam

from active_learning.train.train_model import ModelTrian
from active_learning.train.dp_kpu import ModelKPU
from active_learning.explore.run_model_md import Explore
from active_learning.label.labeling import Labeling

from active_learning.init_bulk.init_bulk_run import init_bulk_run
def run_iter():
    system_info = json.load(open(sys.argv[2]))
    machine_info = json.load(open(sys.argv[3]))
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
    #!. make scf work
    lab.make_scf_work()
    #2. do scf work
    lab.do_labeling()
    #3. post process, collect movement
    lab.post_process_scf()
    
    
def do_training_work(itername:str, resource : Resource, input_param: InputParam):
    mtrain = ModelTrian(itername, resource, input_param)
    # 1. generate feature
    mtrain.generate_feature()
    # 2. do gen_feat job
    mtrain.do_gen_feature_work()
    # 3. create train work dirs
    mtrain.make_train_work()
    # 4. run training job
    mtrain.do_train_job()
    # 5. do post process after training
    mtrain.post_process_train()
    print("{} done !".format("train_model"))

def do_exploring_work(itername:str, resource : Resource, input_param: InputParam):
    md = Explore(itername, resource, input_param)
    # 1. make md work files
    md.make_md_work()
    
    # 2. do md job
    md.do_md_jobs()
    
    # 3. do post process after lammps md running
    md.post_process_md()
    
    # 4. select images
    if input_param.strategy.uncertainty == UNCERTAINTY.committee:
        md.select_image_by_committee()
        # committee: read model deviation file under md file
    elif input_param.strategy.uncertainty == UNCERTAINTY.kpu:
        uncertainty_analyse_kpu(itername, resource, input_param)

def uncertainty_analyse_kpu(itername:str, resource : Resource, input_param: InputParam):
    mkpu = ModelKPU(itername, resource, input_param)
    # 1. make kpu work dirs
    mkpu.make_kpu_work()
    # 2. do kpu job
    mkpu.do_kpu_jobs()
    # 3. post process after kpu calculate: select images
    mkpu.post_process_kpu()

def init_bulk():
    system_info = json.load(open(sys.argv[2]))
    machine_info = json.load(open(sys.argv[3]))
    resource = Resource(machine_info, job_type=AL_WORK.init_bulk)
    input_param = InitBulkParam(system_info)
    os.chdir(input_param.root_dir)
    
    print("The work dir change to {}".format(os.getcwd()))
    init_bulk_run(resource, input_param)
    print("Init Bulk Work Done!")

def print_init_json_template():
    pass

def init_surface():
    pass

def print_run_json_template():
    pass

def print_cmd():
    pass

def main():
    if "init_bulk".upper() in sys.argv[1].upper():
        init_bulk()

    elif "int_surface".upper() in sys.argv[1].upper():
        init_surface()

    elif "run".upper() in sys.argv[1].upper():
        run_iter()

    elif "init_json".upper() in sys.argv[1].upper():
        print_init_json_template()

    elif "run_json".upper() in sys.argv[1].upper():
        print_run_json_template()
    
    elif "-h".upper() in sys.argv[1].upper() or "help".upper() in sys.argv[1].upper():
        print_cmd()

if __name__ == "__main__":
    main()
