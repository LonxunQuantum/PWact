import os, sys
import json
# import argparse

from utils.format_input_output import make_iter_name
from utils.file_operation import write_to_file

from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam

from active_learning.train.train_model import ModelTrian
from active_learning.explore.run_model_md import PWmat_MD
from active_learning.labeling import Labeling
from utils.separate_movement import MovementOp

def run_iter():
    system_info = json.load(open(sys.argv[2]))
    machine_info = json.load(open(sys.argv[3]))
    resouce = Resource(machine_info)
    input_param = InputParam(system_info)
    cwd = os.getcwd()
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

    cont = True
    ii = -1
    numb_task = 4
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
                do_training_work(iter_name, resouce, input_param)
            elif jj == 1:
                print ("exploring start: iter {} - task {}".format(ii, jj))
                do_exploring_work(iter_name, resouce, input_param)
            elif jj == 2:
                print ("uncertainty analyse (kpu): iter {} - task {}".format(ii, jj))
                uncertainty_analyse(iter_name) #exploring/kpu_dir
            elif jj == 3:
                print ("run_fp: iter {} - task {}".format(ii, jj))
                run_fp(iter_name)
            #record_iter
            write_to_file(record, "\n{} {}".format(ii, jj))

def run_fp(itername):
    lab = Labeling(itername)
    lab.do_labeling()

def do_exploring_work(itername:str, resouce : Resource, param_input: InputParam):
    md = PWmat_MD(itername)
    #do pwmat+dpkf md
    if "slurm" in md.system_info.keys():
        md.dpkf_md_slurm()
    else:
        md.dpkf_md()
    print("{} done !".format("pwmat dpkf md_run"))
    
    #separate the MOVEMENT file to single image
    movement_path = os.path.join(md.work_dir.md_dir, "MOVEMENT")
    atom_config_save_dir = md.work_dir.md_traj_dir
    mop = MovementOp(movement_path)
    if os.path.exists(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT")) is False:
        mop.save_all_image_as_one_movement(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT"), md.out_gap)
    mop = MovementOp(os.path.join(md.work_dir.md_dpkf_dir, "MOVEMENT"))
    mop.save_each_image_as_atom_config(atom_config_save_dir) # #md_traj_dir
    print("{} done !".format("movement separates to trajs"))

    md.convert2dpinput()
    # md.separate_train_dir()
    print("{} done !".format("convert2dpinput"))

def uncertainty_analyse(itername):
    mtrain = ModelTrian(itername)
    mtrain.make_kpu()
    print("{} done !".format("calculate kpu"))

def do_training_work(itername:str, resouce : Resource, param_input: InputParam):
    mtrain = ModelTrian(itername, resouce, param_input)
    # 1. generate feature
    mtrain.generate_feature()
    # 2. do gen_feat job
    mtrain.do_gen_feature_work()
    # 3. create train work dirs
    mtrain.make_train_work()
    # 4. run training job
    mtrain.do_train_job()
    print("{} done !".format("train_model"))
    
# def test():
    # cwd = os.getcwd()
    # stdpath = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/iter.0000/exploring/md_dpkf_dir"
    # os.chdir(stdpath)
    # import subprocess
    # # result = subprocess.call("bash -i gen_dpkf_data.sh", shell=True)
    # res = os.popen("bash -i gen_dpkf_data.sh")
    # # assert(result == 0)
    # print(res.readlines())

def init_bulk():
    pass

def init_surface():
    pass

def main():
    if "init_bulk".upper() in sys.argv[1].upper():
        init_bulk()

    elif "int_surface".upper() in sys.argv[1].upper():
        init_surface()

    elif "run".upper() in sys.argv[1].upper():
        run_iter()
    # test()

if __name__ == "__main__":
    main()
