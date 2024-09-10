import os
import glob

from pwact.active_learning.init_bulk.relax import Relax
from pwact.active_learning.init_bulk.duplicate_scale import do_pertub_work, do_post_pertub, pertub_done, \
    duplicate_scale_done, duplicate_scale, do_post_duplicate_scale
from pwact.active_learning.init_bulk.aimd import AIMD
from pwact.active_learning.init_bulk.relabel import Relabel
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.slurm.slurm import scancle_job
from pwact.utils.constant import INIT_BULK, DFT_STYLE, TEMP_STRUCTURE
from pwact.utils.file_operation import copy_file, copy_dir, search_files, del_file, del_file_list, write_to_file
from pwact.data_format.configop import extract_pwdata

def init_bulk_run(resource: Resource, input_param:InitBulkParam):
    #1. do relax
    if input_param.is_relax:
        relax = Relax(resource, input_param)
        if not relax.check_work_done():
            # make relax work dir
            relax.make_relax_work()
            # do relax jobs
            relax.do_relax_jobs()
            # do post process
        relax.do_post_process()
    # do super cell and scale
    if not duplicate_scale_done(input_param):
        duplicate_scale(resource, input_param)
        do_post_duplicate_scale(resource, input_param)

    # do pertub
    if not pertub_done(input_param):
        do_pertub_work(resource, input_param)
        do_post_pertub(resource, input_param)

    # do aimd
    if input_param.is_aimd:
        aimd = AIMD(resource, input_param)
        if not aimd.check_work_done():
            aimd.make_aimd_work()
            aimd.do_aimd_jobs()
            aimd.do_post_process()

    # do relabel
    if input_param.is_scf:
        relabel = Relabel(resource, input_param)
        if not relabel.check_work_done():
            relabel.make_scf_work()
            relabel.do_scf_jobs()
            relabel.do_post_process()
    do_collection(resource, input_param)
       
def do_collection(resource: Resource, input_param:InitBulkParam):
    init_configs = input_param.sys_config
    
    relax_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
    pertub_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
    aimd_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.aimd)
    collection_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.collection)
    
    real_collection_dir = os.path.join(input_param.root_dir, INIT_BULK.collection)
    real_relax_dir = os.path.join(input_param.root_dir, INIT_BULK.relax)
    real_duplicate_scale_dir = os.path.join(input_param.root_dir, INIT_BULK.super_cell_scale) 
    real_pertub_dir = os.path.join(input_param.root_dir, INIT_BULK.pertub)
    real_aimd_dir = os.path.join(input_param.root_dir, INIT_BULK.aimd)

    relabel_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.scf)
    real_relabel_dir = os.path.join(input_param.root_dir, INIT_BULK.scf)
    
        
    for init_config in init_configs:
        init_config_name = "init_config_{}".format(init_config.config_index)
        
        collection_work_dir = os.path.join(collection_dir, init_config_name)
        if not os.path.exists(collection_work_dir):
            os.makedirs(collection_work_dir)
        #1. copy relax config file
        if init_config.relax:
            source_config = os.path.join(relax_dir, init_config_name, DFT_STYLE.get_relaxed_config(resource.dft_style))
            target_config = os.path.join(collection_work_dir, DFT_STYLE.get_relaxed_config(resource.dft_style))
            copy_file(source_config, target_config)
        #2. copy super cell and scaled config file
        config_template = "*{}".format(DFT_STYLE.get_postfix(resource.dft_style))
        source_file_list = search_files(os.path.join(duplicate_scale_dir, init_config_name), config_template)
        for file in source_file_list:
            target_file = os.path.join(collection_dir, init_config_name, os.path.basename(file))
            copy_file(file, target_file)
                    
        #3. copy perturb structure
        if init_config.perturb is not None:
            source_dir = search_files(os.path.join(pertub_dir, init_config_name), "*")
            for source in source_dir:
                # copy .../path/pertub/init_config_0/0.95_scale to .../path/collection/init_config_0/0.95_scale
                target_dir = os.path.join(collection_dir, init_config_name, "{}_{}".format(os.path.basename(source),INIT_BULK.pertub))
                copy_dir(source, target_dir, symlinks=False)
            
        #4. copy aimd result
        source_aimd = []
        if init_config.aimd is True:
            source_aimd = search_files(os.path.join(aimd_dir, init_config_name), "*/*{}/{}".format(INIT_BULK.aimd, DFT_STYLE.get_aimd_config(resource.dft_style)))
            if len(source_aimd) == 0:
                continue
            source_aimd = sorted(source_aimd)
            #5. convert the aimd files (for vasp is outcar, for pwmat is movement) to npy format
            extract_pwdata(data_list=source_aimd, 
                    data_format=DFT_STYLE.get_aimd_config_format(resource.dft_style),
                    datasets_path=os.path.join(collection_dir, init_config_name, INIT_BULK.npy_format_save_dir), 
                    train_valid_ratio=input_param.train_valid_ratio, 
                    data_shuffle=input_param.data_shuffle, 
                    merge_data=True,
                    interval=1
                )

        #6 convert relabel datas
        if init_config.scf:
            source_scf = search_files(os.path.join(relabel_dir, init_config_name),\
                 "*/*/*/{}".format(DFT_STYLE.get_aimd_config(resource.scf_style)))
                 # init/0-aimd/0-scf/OUTCAR
            if len(source_aimd) == 0:
                continue
            source_scf = sorted(source_scf, key=lambda x:int(os.path.basename(os.path.dirname(x)).split('-')[0]), reverse=False)
            #5. convert the aimd files (for vasp is outcar, for pwmat is movement) to npy format
            extract_pwdata(data_list=source_scf, 
                    data_format=DFT_STYLE.get_format_by_postfix(os.path.basename(source_scf[0])),
                    datasets_path=os.path.join(collection_dir, init_config_name, "scf_pwdata"), 
                    train_valid_ratio=input_param.train_valid_ratio, 
                    data_shuffle=input_param.data_shuffle, 
                    merge_data=True,
                    interval=1
                )

    # delete link files
    del_file(real_relax_dir)
    del_file(real_duplicate_scale_dir)
    del_file(real_pertub_dir)
    del_file(real_aimd_dir)
    del_file(real_relabel_dir)

    # copy collection file to target
    copy_dir(collection_dir, real_collection_dir)
    if not input_param.reserve_work:
        # mv collection to real dir
        temp_work_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir)
        del_file_list([temp_work_dir])

    # print the dir of pwdatas from aimd
    pwdatas = search_files(real_collection_dir, "*/{}".format(INIT_BULK.npy_format_save_dir))
    if len(pwdatas) > 0:
        pwdatas = sorted(pwdatas)
        result_lines = ["\"{}\",".format(_) for _ in pwdatas]
        result_lines = "\n".join(result_lines)
        # result_lines = result_lines[:-1] # Filter the last ','
        result_save_path = os.path.join(real_collection_dir, INIT_BULK.npy_format_name)
        write_to_file(result_save_path, result_lines, mode='w')
    
    # print the dir of relabel_pwdatas from relabel
    relebel_datas = search_files(real_collection_dir, "*/{}".format("scf_pwdata"))
    if len(relebel_datas) > 0:
        pwdatas = sorted(relebel_datas)
        result_lines = ["\"{}\",".format(_) for _ in pwdatas]
        result_lines = "\n".join(result_lines)
        # result_lines = result_lines[:-1] # Filter the last ','
        result_save_path = os.path.join(real_collection_dir, "scf_pwdata")
        write_to_file(result_save_path, result_lines, mode='w')

def scancel_jobs(work_dir):
    relax_jobs = glob.glob(os.path.join(work_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax, "slurm-*.out"))
    scf_jobs = glob.glob(os.path.join(work_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.aimd, "slurm-*.out"))
    if len(scf_jobs) > 0:
        scancle_job(scf_jobs)
    elif len(relax_jobs) > 0:
        scancle_job(relax_jobs)