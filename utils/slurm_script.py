import glob
import os
from math import ceil
from utils.constant import DFT_STYLE
GPU_SCRIPT_HEAD = \
"#!/bin/sh\n\
#SBATCH --job-name={}\n\
#SBATCH --nodes={}\n\
#SBATCH --ntasks-per-node={}\n\
#SBATCH --gres=gpu:{}\n\
#SBATCH --gpus-per-task={}\n\
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

CONDA_ENV = '__conda_setup="$(\'/data/home/wuxingxing/anaconda3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)"\n' \
       'if [ $? -eq 0 ]; then\n' \
       '    eval "$__conda_setup"\n' \
       'else\n' \
       '    if [ -f "/data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh" ]; then\n' \
       '        . "/data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh"\n' \
       '    else\n' \
       '        export PATH="/data/home/wuxingxing/anaconda3/bin:$PATH"\n' \
       '    fi\n' \
       'fi\n' \
       'unset __conda_setup\n' \
       '# <<< conda initialize <<<\n' \
       'conda activate torch2_feat\n\n'

'''
Description: 
Obtain the execution status of the slurm jobs under the dir:
param {*} dir
Returns: 
Author: WU Xingxing
'''
def get_slurm_job_run_info(dir:str, job_patten:str="*.job", tag_patten:str="tag.*.success"):
    slurm_job_files = glob.glob(os.path.join(dir, job_patten))
    slrum_job_dirs = [os.path.dirname(_) for _ in slurm_job_files]

    slurm_tag_sucess_files = glob.glob(os.path.join(dir, tag_patten))
    slrum_tag_sucess_dirs = [os.path.dirname(_) for _ in slurm_tag_sucess_files]

    slurm_failed = []
    slurm_success = []

    for i, d in enumerate(slrum_job_dirs):
        if d in slrum_tag_sucess_dirs:
            slurm_success.append(slurm_job_files[i])
        else:
            slurm_failed.append(slurm_job_files[i])
            
    return slurm_failed, slurm_success


'''
description: 
    split the job_list with groupsize
param {int} groupsize
param {list} job_list: N jobs
return {*} [["job1","job2",...,"job_groupseze"], ..., [..., "job_N", "NONE",...,"NONE"]]
author: wuxingxing
'''
def split_job_for_group(groupsize:int , job_list:list[str], parallel_num=1):
    groupsize = 1 if groupsize is None else groupsize
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
                            source_list,
                            module_list,
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
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, gpu_per_node, 1, queue_name)
        
        for custom_flag in custom_flags:
            script += custom_flag + "\n"
                
        # set conda env
        script += "\n"
        script += CONDA_ENV
        script += "\n"        
        script += "start=$(date +%s)\n"

        # set source_list
        for source in source_list:
            script += source + "\n"
        script += "\n"        
        # set module_list
        for module in module_list:
            script += "module load " + module + "\n"
        script += "\n"
        
        job_cmd = ""
        job_id = 0
        job_tag_list = []
        
        if check_type == DFT_STYLE.pwmat:
            check_info = pwmat_check_success(task_tag, task_tag_faild)
        else:
            check_info = common_check_success(task_tag, task_tag_faild)
            
        while job_id < len(group):
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
            job_cmd += "wait\n\n"
        
        script += job_cmd
        
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
