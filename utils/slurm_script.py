import glob
import os

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
       'conda activate torch2\n\n'

'''
description: 
    Set the basic app for Slurm job dependency
param {list} custom_flags
param {list} source_list
param {list} module_list
return {*}
author: wuxingxing
'''
def set_slurm_comm_basis(custom_flags:list[str]=[], source_list:list[str]=[], module_list:list[str]=[]):
    script = ""
    # set custom_flags
    for custom_flag in custom_flags:
        script += custom_flag + "\n"
    script += "\n"
    # set source_list
    for source in source_list:
        script += source + "\n"
    script += "\n"        
    # set module_list
    for module in module_list:
        script += "module load " + module + "\n"
    script += "\n"
    return script

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
def set_slurm_script_content(gpu_per_node, 
                             number_node, 
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
                             parallel_num:int=1
                             ):
        # set head
        script = ""
        if gpu_per_node is None:
            script += CPU_SCRIPT_HEAD.format(job_name, number_node, cpu_per_node, queue_name)
        else:
            script += GPU_SCRIPT_HEAD.format(job_name, number_node, gpu_per_node, gpu_per_node, 1, queue_name)
        
        script += set_slurm_comm_basis(custom_flags, source_list, module_list)
        
        # set conda env
        script += "\n"
        script += CONDA_ENV
        script += "\n"        
        script += "start=$(date +%s)\n"

        job_cmd = ""
        job_id = 0
        job_tag_list = []
        while job_id < len(group):
            for i in range(parallel_num):
                if group[job_id] == "NONE":
                    job_id += 1
                    continue
                job_cmd += "{\n"
                job_cmd += "cd {}\n".format(group[job_id])
                job_cmd += "if [ ! -f {} ] ; then\n".format(task_tag)
                job_cmd += "    {}\n".format(run_cmd_template)
                job_cmd += "    if test $? -eq 0; then touch {}; else touch {}; fi\n".format(task_tag, task_tag_faild)
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