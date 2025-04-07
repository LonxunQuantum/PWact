import os
import glob

from pwact.active_learning.init_bulk.relax import Relax
from pwact.active_learning.init_bulk.duplicate_scale import do_pertub_work, do_post_pertub, pertub_done, \
    duplicate_scale_done, duplicate_scale, do_post_duplicate_scale
from pwact.active_learning.init_bulk.aimd import AIMD
from pwact.active_learning.init_bulk.explore import BIGMODEL
from pwact.active_learning.init_bulk.relabel import Relabel
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam
from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.slurm.slurm import scancle_job
from pwact.utils.constant import INIT_BULK, DFT_STYLE, TEMP_STRUCTURE, PWDATA, LABEL_FILE_STRUCTURE
from pwact.utils.file_operation import copy_file, copy_dir, search_files, del_file, del_file_list, write_to_file, del_file_list_by_patten
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

    elif input_param.is_bigmodel:
        bigmodel = BIGMODEL(resource, input_param)
        if not bigmodel.check_work_done():
            bigmodel.make_bigmodel_work()
            bigmodel.do_bigmodel_jobs()
            bigmodel.do_post_process()
            
        # do direct
        if not bigmodel.check_direct_done():
            bigmodel.make_direct_work()
            bigmodel.do_direct_jobs() #  after 

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
    bigmodel_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.bigmodel)
    direct_dir = os.path.join(bigmodel_dir, INIT_BULK.direct)
    
    real_collection_dir = os.path.join(input_param.root_dir, INIT_BULK.collection)
    real_relax_dir = os.path.join(input_param.root_dir, INIT_BULK.relax)
    real_duplicate_scale_dir = os.path.join(input_param.root_dir, INIT_BULK.super_cell_scale) 
    real_pertub_dir = os.path.join(input_param.root_dir, INIT_BULK.pertub)
    real_aimd_dir = os.path.join(input_param.root_dir, INIT_BULK.aimd)

    relabel_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.scf)
    real_relabel_dir = os.path.join(input_param.root_dir, INIT_BULK.scf)
    real_bigmodel_dir = os.path.join(input_param.root_dir, INIT_BULK.bigmodel)
    real_direct_dir = os.path.join(input_param.root_dir, INIT_BULK.direct)
     
    result_save_path = []
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
            extract_pwdata(input_data_list=source_aimd, 
                            intput_data_format= DFT_STYLE.get_aimd_config_format(resource.dft_style),
                            save_data_path  = os.path.join(collection_dir, init_config_name),
                            save_data_name  = INIT_BULK.get_save_format(input_param.data_format), 
                            save_data_format= input_param.data_format,
                            data_shuffle=input_param.data_shuffle, 
                            interval=1
                )
            result_save_path.append(os.path.join(real_collection_dir, init_config_name, INIT_BULK.get_save_format(input_param.data_format)))
                
    #6 convert relabel datas
    if input_param.is_scf:
        source_scf = search_files(relabel_dir,\
                "*/{}".format(DFT_STYLE.get_scf_config(resource.dft_style)))
                # init/0-aimd/0-scf/OUTCAR
        source_scf = sorted(source_scf, key=lambda x:int(os.path.basename(os.path.dirname(x))), reverse=False)
        #5. convert the aimd files (for vasp is outcar, for pwmat is movement) to npy format
        extract_pwdata(input_data_list=source_scf, 
                intput_data_format= DFT_STYLE.get_format_by_postfix(os.path.basename(source_scf[0])),
                save_data_path  = os.path.join(collection_dir, INIT_BULK.scf), 
                save_data_name  = INIT_BULK.get_save_format(input_param.data_format), 
                save_data_format= input_param.data_format,
                data_shuffle=input_param.data_shuffle, 
                interval=1
            )
        result_save_path.append(os.path.join(real_collection_dir, INIT_BULK.scf, INIT_BULK.get_save_format(input_param.data_format)))
    #7 bigmodel infos
    if input_param.is_bigmodel:
        if not input_param.is_scf:
            extract_pwdata(input_data_list=[os.path.join(direct_dir, INIT_BULK.direct_traj)], 
                    intput_data_format= PWDATA.extxyz,
                    save_data_path  = direct_dir, 
                    save_data_name  = INIT_BULK.get_save_format(input_param.data_format), 
                    save_data_format= input_param.data_format,
                    data_shuffle=input_param.data_shuffle, 
                    interval=1
                )
            result_save_path.append(os.path.join(real_collection_dir, INIT_BULK.bigmodel, INIT_BULK.direct, INIT_BULK.get_save_format(input_param.data_format)))
        # copy bigmodel and direct files to realdir
        copy_dir(bigmodel_dir, os.path.join(collection_dir, INIT_BULK.bigmodel))        
    
    if len(result_save_path) > 0:
        _path_path = []
        for _data_path in result_save_path:
            if input_param.data_format == PWDATA.extxyz:
                _path_path.append(_data_path)
            elif input_param.data_format == PWDATA.pwmlff_npy: # */PWdata/*.npy
                tmp = search_files(_data_path, "*/position.npy")
                _path_path.extend([os.path.dirname(_) for _ in tmp])
            
        result_lines = ["\"{}\",".format(_) for _ in _path_path]
        result_lines = "\n".join(result_lines)
        # result_lines = result_lines[:-1] # Filter the last ','
        result_save_path = os.path.join(collection_dir, INIT_BULK.npy_format_name)
        write_to_file(result_save_path, result_lines, mode='w')

    # delete link files
    del_file(real_relax_dir)
    del_file(real_duplicate_scale_dir)
    del_file(real_pertub_dir)
    del_file(real_aimd_dir)
    del_file(real_relabel_dir)
    del_file(real_bigmodel_dir)
    # del slurm logs and tags
    del_file_list_by_patten(os.path.join(collection_dir, INIT_BULK.bigmodel), "slurm-*")
    del_file_list_by_patten(os.path.join(collection_dir, INIT_BULK.bigmodel, INIT_BULK.direct), "slurm-*")
    # copy collection file to target
    copy_dir(collection_dir, real_collection_dir)
    if not input_param.reserve_work:
        # mv collection to real dir
        temp_work_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir)
        del_file_list([temp_work_dir])


def scancel_jobs(work_dir):
    relax_job = os.path.join(work_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
    scf_job = os.path.join(work_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.aimd)
    print("Cancel aimd task:")
    scancle_job(scf_job)
    print("Cancel relax task:")
    scancle_job(relax_job)