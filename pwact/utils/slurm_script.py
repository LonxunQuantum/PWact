import glob
import re
import os
from math import ceil
from pwact.utils.constant import DFT_STYLE
GPU_SCRIPT_HEAD = \
"#!/bin/sh\n\
#SBATCH --job-name={}\n\
#SBATCH --nodes={}\n\
#SBATCH --ntasks-per-node={}\n\
#SBATCH --gres=gpu:{}\n\
#SBATCH --partition={}\n\
\
"

CPU_SCRIPT_HEAD = \
"#!/bin/sh\n\
#SBATCH --job-name={}\n\
#SBATCH --nodes={}\n\
#SBATCH --ntasks-per-node={}\n\
#SBATCH --partition={}\n\
\
"

'''
Description: 
Obtain the execution status of the slurm jobs under the 'dir'
    the slurm.job and tag format are number-jobname.job, number-tag-type-success.
    for example:
        iter.0000/explore/md/
        0-md.job 1-md.job 2-md.job 3-md.job 4-md.job 5-md.job 
        0-tag.md.success 1-tag.md.success 2-tag.md.success 3-tag.md.success 4-tag.md.success 5-tag.md.success

        or

        0-scf.job 1-scf.job 2-scf.job 3-scf.job 4-scf.job
        0-tag.scf.success 1-tag.scf.success 2-tag.scf.success 3-tag.scf.success 4-tag.scf.success
    new change:
        only jugt the slurm file is done by the tag under each subwork tag. so the tag of job file is nouse maybe
param {*} dir
Returns: 
Author: WU Xingxing
'''
def get_slurm_job_run_info(dir:str, job_patten:str="*.job", tag_patten:str="tag.*.success", for_back:bool=False):
    slurm_job_files = sorted(glob.glob(os.path.join(dir, job_patten)))
    slrum_job_dirs = [int(os.path.basename(_).split('-')[0]) for _ in slurm_job_files]

    slurm_tag_sucess_files = sorted(glob.glob(os.path.join(dir, tag_patten)))
    slrum_tag_sucess_dirs = [int(os.path.basename(_).split('-')[0]) for _ in slurm_tag_sucess_files]
    
    slurm_failed = []
    slurm_success = []

    if for_back is False:
        for slurm_file in slurm_job_files:
            if slurm_job_done_by_jobtag(slurm_file):
                slurm_success.append(slurm_file)
            else:
                slurm_failed.append(slurm_file)
    else:
        for i, d in enumerate(slrum_job_dirs):
            if d in slrum_tag_sucess_dirs:
                slurm_success.append(slurm_job_files[i])
            else:
                slurm_failed.append(slurm_job_files[i])
            
    return slurm_failed, slurm_success


def slurm_job_done_by_jobtag(slurm_file):
    with open(slurm_file, 'r') as f:
        content = f.read()
    cd_pattern = r'cd\s+([^\n]+)'
    directories = re.findall(cd_pattern, content)        
    if not directories:
        raise Exception("Error! There is no task in the slurm.job file {}".format(slurm_file))
    for directory in directories:
        directory = directory.strip()
        success_file = glob.glob(os.path.join(directory, "*.success"))
        if len(success_file) > 0:
            continue
        else:
            return False
    return True


def recheck_slurm_by_jobtag(slurm_files:list[str], tag):
    remain_job = []
    for slurm_file in slurm_files:
        with open(slurm_file, 'r') as f:
            content = f.read()
        cd_pattern = r'cd\s+([^\n]+)'
        directories = re.findall(cd_pattern, content)        
        if not directories:
            raise Exception("Error! There is no task in the slurm.job file {}".format(slurm_file))
        for directory in directories:
            directory = directory.strip()
            success_file = os.path.join(directory, tag)
            if os.path.exists(success_file):
                continue
            else:
                remain_job.append(slurm_file)
                break
    return remain_job

# def slurm_job_is_done_by_jobtag(dir:str, job_patten:str="*.job", tag_patten:str="tag.*.success"):
#     slurm_job_files = sorted(glob.glob(os.path.join(dir, job_patten)))
#     slurm_failed = []
#     for slurm_file in slurm_job_files:
#         with open(slurm_file, 'r') as f:
#             content = f.read()
#         cd_pattern = r'cd\s+([^\n]+)'
#         directories = re.findall(cd_pattern, content)        
#         if not directories:
#             raise Exception("Error! There is no task in the slurm.job file {}".format(slurm_file))
#         for directory in directories:
#             directory = directory.strip()
#             success_file = os.path.join(directory, tag_patten)
#             if os.path.exists(success_file):
#                 continue
#             else:
#                 slurm_failed.append(slurm_file)
#                 break
#     return slurm_failed


'''
description: 
    split the job_list with groupsize
param {int} groupsize
param {list} job_list: N jobs
return {*} [["job1","job2",...,"job_groupseze"], ..., [..., "job_N", "NONE",...,"NONE"]]
author: wuxingxing
'''
def split_job_for_group(groupsize:int , job_list:list[str], parallel_num=1):
    groupsize = len(job_list) if groupsize == -1 else groupsize
    if groupsize > 1:
        groupsize_adj = ceil(groupsize/parallel_num)
        if groupsize_adj*parallel_num > groupsize:
            groupsize_adj = ceil(groupsize/parallel_num)*parallel_num
            print("the groupsize automatically adjusts from {} to {}".format(groupsize, groupsize_adj))
            groupsize = groupsize_adj
            
    group_num = len(job_list) // groupsize
    sub_list = []
    for i in range(group_num):
        sub_list.append(job_list[i*groupsize:(i+1)*groupsize])
    
    last_sub = []
    if len(sub_list)*groupsize < len(job_list):
        for sub_index in range(len(sub_list)*groupsize, len(job_list)):
            last_sub.append(job_list[sub_index])
        while len(last_sub) < groupsize:
            last_sub.append("NONE")
        sub_list.append(last_sub)
    return sub_list

def get_job_tag_check_string(job_tags:list[str], true_script:str="", error_script:str=""):
    script = "if "
    for index, job_tag in enumerate(job_tags):
        script += "[ -f {} ]".format(job_tag)
        if index < len(job_tags)-1:
            script += " && "
    script += "; then\n"
    script += true_script
    script += "else\n"
    script += error_script
    script += "fi\n"
    return script


'''
description: 
make slurm job content
   run_cmd_template = "mpirun -np {} PWmat".format(cpu_per_node or gpu_per_node)
return {*}
author: wuxingxing
'''
def set_slurm_script_content(
                            number_node, 
                            gpu_per_node, #None
                            cpu_per_node,
                            queue_name,
                            custom_flags,
                            env_script,
                            job_name,
                            run_cmd_template,
                            group:list[str],
                            job_tag:str,
                            task_tag:str,
                            task_tag_faild:str,
                            parallel_num:int=1,
                            check_type:str=None
                            ):
        # set head
        script = ""
        if gpu_per_node is None or gpu_per_node == 0:
            script += CPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, queue_name)
            # script += "export CUDA_VISIBLE_DEVICES=''\n"
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, gpu_per_node, queue_name)
        
        for custom_flag in custom_flags:
            script += custom_flag + "\n"
        
        # set conda env
        script += "\n"
        # script += "\n"

        script += "echo \"SLURM_SUBMIT_DIR is $SLURM_SUBMIT_DIR\"\n\n"
        script += "echo \"Starting job $SLURM_JOB_ID at \" `date`\n\n"
        script += "echo \"Running on nodes: $SLURM_NODELIST\"\n\n"

        script += "start=$(date +%s)\n"

        # set source_list
        script += env_script
        script += "\n"
        
        job_cmd = ""
        job_id = 0
        job_tag_list = []
        
        if check_type == DFT_STYLE.pwmat:
            check_info = pwmat_check_success(task_tag, task_tag_faild)
        else:
            check_info = common_check_success(task_tag, task_tag_faild)
            
        while job_id < len(group):
            if group[job_id] == "NONE":
                job_id += 1
                continue
            
            for i in range(parallel_num):
                if group[job_id] == "NONE":
                    job_id += 1
                    continue
                job_cmd += "{\n"
                job_cmd += "cd {}\n".format(group[job_id])
                job_cmd += "if [ ! -f {} ] ; then\n".format(task_tag)
                job_cmd += "    {}\n".format(run_cmd_template)
                job_cmd += check_info
                job_cmd += "fi\n"
                job_cmd += "} &\n\n"
                job_tag_list.append(os.path.join(group[job_id], task_tag))
                job_id += 1
                if job_id > len(group):
                    break
            job_cmd += "wait\n\n"
        
        script += job_cmd
        script += "echo \"Job $SLURM_JOB_ID done at \" `date`\n\n"

        right_script = ""
        right_script += "    end=$(date +%s)\n"
        right_script += "    take=$(( end - start ))\n"
        right_script += "    echo Time taken to execute commands is ${{take}} seconds > {}\n".format(job_tag)
        right_script += "    exit 0\n"   
        error_script  = "    exit 1\n"
        
        script += get_job_tag_check_string(job_tag_list, right_script, error_script)


        return script
    

def pwmat_check_success(task_tag:str, task_tag_faild:str):
    script  = ""
    script += "    if [ -f REPORT ]; then\n"
    script += "        last_line=$(tail -n 1 REPORT)\n"
    script += "        if [[ $last_line == *\"time\"* ]]; then\n"
    script += "            touch {}\n".format(task_tag)
    script += "        else\n"
    script += "            touch {}\n".format(task_tag_faild)
    script += "        fi\n"
    script += "    else\n"
    script += "        touch {}\n".format(task_tag_faild)
    script += "    fi\n"
    return script

def common_check_success(task_tag:str, task_tag_failed:str):
    script = ""
    script += "    if test $? == 0; then\n"
    script += "        touch {}\n".format(task_tag)
    script += "    else\n"
    script += "        touch {}\n".format(task_tag_failed)
    script += "    fi\n"
    return script
