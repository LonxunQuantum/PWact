import os

from active_learning.init_bulk.relax import Relax
from active_learning.init_bulk.duplicate_scale import do_pertub, do_post_pertub, pertub_done, \
    duplicate_scale_done, duplicate_scale, do_post_duplicate_scale
from active_learning.init_bulk.aimd import AIMD
from active_learning.user_input.init_bulk_input import InitBulkParam
from active_learning.user_input.resource import Resource

from utils.constant import INIT_BULK, TEMP_STRUCTURE, PWMAT
from utils.file_operation import merge_files_to_one, copy_file, copy_dir, search_files, file_shell_op, del_file, write_to_file
from utils.gen_format.pwdata import Save_Data

def init_bulk_run(resource: Resource, input_param:InitBulkParam):
    #1. do relax
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
        do_pertub(resource, input_param)
        do_post_pertub(resource, input_param)
    # do aimd
    aimd = AIMD(resource, input_param)
    if not aimd.check_work_done():
        aimd.make_aimd_work()
        aimd.do_aimd_jobs()
        aimd.do_post_process()
    # do collection
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
    
    pwdata_list = []
    for init_config in init_configs:
        init_config_name = "init_config_{}".format(init_config.config_index)
        
        collection_work_dir = os.path.join(collection_dir, init_config_name)
        if not os.path.exists(collection_work_dir):
            os.makedirs(collection_work_dir)
        #1. copy relax atom.config
        if init_config.relax:
            source_config = os.path.join(relax_dir, init_config_name, INIT_BULK.final_config)
            target_config = os.path.join(collection_work_dir, INIT_BULK.realx_config)
            copy_file(source_config, target_config)
        
        config_template = "*{}".format(PWMAT.config_postfix)
        source_file_list = search_files(os.path.join(duplicate_scale_dir, init_config_name), config_template)
        for file in source_file_list:
            target_file = os.path.join(collection_dir, init_config_name, os.path.basename(file))
            copy_file(file, target_file)
                    
        #2. copy perturb structure
        if init_config.perturb is not None:
            source_dir = search_files(os.path.join(pertub_dir, init_config_name), "*")
            for source in source_dir:
                # copy .../path/pertub/init_config_0/0.95_scale to .../path/collection/init_config_0/0.95_scale
                target_dir = os.path.join(collection_dir, init_config_name, "{}_{}".format(os.path.basename(source),INIT_BULK.pertub))
                copy_dir(source, target_dir)
            
        #3. copy aimd result to movements
        target_mvm = ""
        if init_config.aimd is True:
            source_aimd = search_files(os.path.join(aimd_dir, init_config_name), "*/*{}/{}".format(INIT_BULK.aimd, PWMAT.MOVEMENT))
            source_aimd = sorted(source_aimd)
            mvm_save_dir = os.path.join(collection_dir, init_config_name)
            target_mvm = os.path.join(mvm_save_dir, "{}_{}".format(PWMAT.MOVEMENT, INIT_BULK.aimd.upper()))
            merge_files_to_one(source_aimd, target_mvm)
            
        #4. convert the mvm files to npy format
        if os.path.exists(target_mvm):
            Save_Data(data_path=target_mvm, 
                datasets_path=os.path.join(mvm_save_dir, INIT_BULK.npy_format_save_dir), 
                train_ratio = input_param.train_valid_ratio, 
                random = input_param.data_shuffle, 
                format=PWMAT.MOVEMENT_low)
            pwdata_list.append(os.path.join(mvm_save_dir, INIT_BULK.npy_format_save_dir, os.path.basename(target_mvm)))
    # delete link files
    del_file(real_relax_dir)
    del_file(real_duplicate_scale_dir)
    del_file(real_pertub_dir)
    del_file(real_aimd_dir)

    # delete no use files
    if not input_param.reserve_work:
        # mv collection to real dir
        file_shell_op("mv {} {}".format(collection_dir, real_collection_dir))
        temp_work_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir)
        file_shell_op("rm {} -rf".format(temp_work_dir))
    
    # print the dir of pwdatas
    pwdatas = search_files(real_collection_dir, "*/{}/{}_*".format(INIT_BULK.npy_format_save_dir, PWMAT.MOVEMENT))
    pwdatas = sorted(pwdatas)
    result_lines = ["\"{}\",".format(_) for _ in pwdatas]
    result_lines = "\n".join(result_lines)
    # result_lines = result_lines[:-1] # Filter the last ','
    result_save_path = os.path.join(real_collection_dir, INIT_BULK.npy_format_name)
    write_to_file(result_save_path, result_lines, mode='w')
    