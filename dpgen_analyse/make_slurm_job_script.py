import os


# self.system_info, self.work_dir.md_dpkf_dir, self.work_dir.gen_feat_success_tag, self.work_dir.gen_feat_path
'''
Description: 
    make slurm.job file to generate feature from movement file
param {*} system_info
param {*} work_dir
param {*} tag
param {*} save_path
Returns: 
Author: WU Xingxing
'''
def make_feature_script_slurm(system_info, work_dir, tag, save_path):
    with open(os.path.join("./template_script_head", "gen_feature.job"), 'r') as rf:
        script_head = rf.readlines()

    # source config.yaml
    yaml_path = system_info["train_config"]["config_yaml"]
    # dstd file path not use
    dstd_dir = os.path.join(system_info["init_data_path"][-1], "train")
    # work script path
    gen_feat_path=system_info["train_config"]["gen_feature_script_path"]

    res = ""
    for i in script_head:
        res += i
    res += "\n"
    res += "python {} ".format(gen_feat_path)
    res += "-c {} ".format(yaml_path)
    res += "-d {} ".format(dstd_dir)
    res += "-w {} ".format(work_dir)
    res += "\n"
    res += "test $? -ne 0 && exit 1\n\n"
    res += "echo 0 > {}\n\n".format(tag)
    with open(save_path, "w") as wf:
        wf.write(res)
    return save_path, tag

def make_feature_script(system_info, work_dir, tag, save_path):
    # source config.yaml
    yaml_path = system_info["train_config"]["config_yaml"]
    # dstd file path
    dstd_dir = os.path.join(system_info["init_data_path"][-1], "train")
    # python script dp_mlff.py path
    gen_feat_path=system_info["train_config"]["gen_feature_script_path"]

    res = "\n"
    res += "#!/bin/bash -l\n"
    res += "conda activate mlff_env \n"
    res += "python {} ".format(gen_feat_path)
    res += "-c {} ".format(yaml_path)
    res += "-d {} ".format(dstd_dir)
    res += "-w {} ".format(work_dir)
    res += "\n"
    res += "test $? -ne 0 && exit 1\n\n"
    res += "echo 0 > {}\n\n".format(tag)
    with open(save_path, "w") as wf:
        wf.write(res)