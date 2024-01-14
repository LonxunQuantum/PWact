import os

from active_learning.init_bulk.relax import Relax
from active_learning.init_bulk.duplicate_scale import do_pertub, do_post_pertub, pertub_done, \
    duplicate_scale_done, duplicate_scale, do_post_duplicate_scale
from active_learning.init_bulk.aimd import AIMD
from active_learning.user_input.init_bulk_input import InitBulkParam
from active_learning.user_input.resource import Resource

from utils.constant import INIT_BULK, TEMP_STRUCTURE, PWMAT
from utils.file_operation import mv_file, copy_file, search_files, merge_files_to_one

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
    do_collection(resource, input_param, relax, aimd)
    # delete no use files
    
    
def do_collection(resource: Resource, input_param:InitBulkParam, relax:Relax, aimd:AIMD):
    init_configs = input_param.sys_config
    
    relax_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
    real_relax_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)

    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
    real_duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 

    pertub_dir = os.path.join(input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
    real_pertub_dir = os.path.join(input_param.root_dir,INIT_BULK.pertub)

    collection_dir = os.path.join(input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.collection)
    real_collection_dir = os.path.join(input_param.root_dir,INIT_BULK.collection)
    
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
        
            source_relax_mvm = search_files(os.path.join(relax_dir, init_config_name), PWMAT.MOVEMENT)
            target_mvm_file = os.path.join(collection_work_dir, "mvm_relax_{}".format(init_config_name))
            copy_file(source_relax_mvm[0], target_mvm_file)
                            
        #2. copy atom.configs after duplicate and scale
        if init_config.perturb is not None:
            source_file_list = search_files(os.path.join(duplicate_scale_dir, init_config_name), "*{}".format(PWMAT.config_postfix))
            for file in source_file_list:
                target_file = os.path.join(collection_dir, init_config_name, os.path.basename(file))
                copy_file(file, target_file)
                    
        #3. copy perturb structure
        if init_config.perturb is not None:
            source_structure_dir = os.path.join(pertub_dir, init_config_name)
            target_dir = os.path.join(collection_dir, init_config_name)
            if not os.path.exists(target_dir):
                mv_file(source_structure_dir, target_dir)
            
        #4. copy aimd result to movements
        source_aimd = search_files(os.path.join(aimd.aimd_dir, init_config_name), "*-{}/{}".format(INIT_BULK.aimd, PWMAT.MOVEMENT))
        for source_mvm in source_aimd:
            target_mvm_file = os.path.join(collection_dir, init_config_name, "mvm_{}".format(init_config_name))
            merge_files_to_one(source_mvm, target_mvm_file)
        
        #5. copy relaxed movement 
