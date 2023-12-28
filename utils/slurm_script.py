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
