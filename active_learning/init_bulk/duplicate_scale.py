import os, subprocess
import shutil
import ase
from ase import io

from utils.constant import PWMAT
from utils.file_operation import copy_file, search_files
from utils.constant import INIT_BULK
from utils.supercell import super_cell, scale_config, super_cell_ase

from active_learning.user_input.resource import Resource
from active_learning.user_input.param_input import InputParam
from active_learning.user_input.init_bulk_input import InitBulkParam, Stage

from matersdk.adalearn.generator.perturbation import BatchPerturbStructure

def duplicate_scale(resource: Resource, input_param:InitBulkParam):
    #work dir
    init_configs = input_param.sys_config
    relax_dir = os.path.join(input_param.root_dir, INIT_BULK.relax)
    for init_config in init_configs:
        init_config_name = "init_config_{}".format(init_config.config_index)
        if init_config.relax:
            config = os.path.join(relax_dir, init_config_name, INIT_BULK.final_config)
        else:
            config = init_config.config
        super_cell_scale_dir = os.path.join(input_param.root_dir, INIT_BULK.super_cell_scale, init_config_name)
        super_cell_config = None
        if init_config.super_cell is not None:
            if not os.path.exists(super_cell_scale_dir):
                os.makedirs(super_cell_scale_dir)
            # super_content, lattic_index = super_cell(init_config.super_cell, config, super_cell_config)
            super_cell_config = os.path.join(super_cell_scale_dir, "{}-{}".format(init_config.config_index, INIT_BULK.super_cell_config))
            super_cell_ase(init_config.super_cell, config, super_cell_config)
            
        if len(init_config.scale) > 0:
            for s_index, scale in enumerate(init_config.scale):
                save_scale_config = os.path.join(super_cell_scale_dir, "{}-{}-{}".format(init_config.config_index, s_index, INIT_BULK.scale_config))
                if not os.path.exists(super_cell_scale_dir):
                    os.makedirs(super_cell_scale_dir)
                if super_cell_config is not None:
                    input_scale_config = super_cell_config
                else:
                    input_scale_config = init_config.config
                scale_config(input_scale_config, scale, save_scale_config)


'''
description: 
param {str} work_dir the dir of atom.config file
param {int} pert_num
param {float} cell_pert_fraction
param {float} atom_pert_distance
return {*}
author: wuxingxing
'''
def do_pertub(resource: Resource, input_param:InitBulkParam):
    #work dir 
    # pert_num:int=50
    # cell_pert_fraction:float=0.03
    # atom_pert_distance:float=0.01
    init_configs = input_param.sys_config
    relax_dir = os.path.join(input_param.root_dir, INIT_BULK.relax)
    super_cell_scale_dir = os.path.join(input_param.root_dir, INIT_BULK.super_cell_scale)
    pertub_dir = os.path.join(input_param.root_dir,INIT_BULK.pertub)
    # make pertub dirs
    for init_config in init_configs:
        if init_config.perturb is None:
            continue
        init_config_name = "init_config_{}".format(init_config.config_index)
        pert_config_list = get_config_files_with_order(super_cell_scale_dir, relax_dir, init_config_name, init_config.config)
        
        for index, config in enumerate(pert_config_list):
            config_from_type = INIT_BULK.get_work_type(work_type=os.path.basename(config))# scale or super_cell or relax or init_config
            work_dir = os.path.join(pertub_dir, init_config_name, "{}-{}".format(config_from_type, INIT_BULK.pertub))
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            tmp_pertub_dir = os.path.join(work_dir, 'tmp')
            if not os.path.exists(tmp_pertub_dir):
                os.makedirs(tmp_pertub_dir)
                
            copy_file(config, os.path.join(tmp_pertub_dir, PWMAT.atom_config))
            cwd = os.getcwd()
            os.chdir(work_dir)
            Perturbed = ['tmp']
            BatchPerturbStructure.batch_perturb(
                Perturbed=Perturbed,
                pert_num=init_config.perturb,
                cell_pert_fraction=init_config.cell_pert_fraction,
                atom_pert_distance=init_config.atom_pert_distance
            )
            #Organize the output files, 'mv structures/*.config ..'
            # subprocess.run(["mv structures/*.config .. && rm structures -rf"], shell = True)
            os.chdir(cwd)

'''
description: 
    get configs:
    the order if pertub/*.configs > *-scaled.config > supercell.config > final.config(relax) > init.config
param {str} super_cell_scale_dir
param {str} relax_dir
param {str} init_config_dirname: the dir made by init_config name
param {str} init_config_path: the init config file path
return {*}
author: wuxingxing
'''
def get_config_files_with_order(super_cell_scale_dir:str, relax_dir:str, init_config_dirname:str, init_config_path:str, pertub_dir:str=None):
    config_list = []
    if pertub_dir is not None:
        config_list = search_files(pertub_dir, "{}/{}/*{}".format(init_config_dirname, INIT_BULK.structures, PWMAT.config_postfix))
    
    if len(config_list) == 0:
    # find scaled configs under super_cell_scale dir
        config_list = search_files(super_cell_scale_dir, "{}/*-{}".format(init_config_dirname, INIT_BULK.scale_config))
    
    if len(config_list) == 0:
    # find super cell config under super_cell_scale dir
        config_list = search_files(super_cell_scale_dir, "{}/*-{}".format(init_config_dirname, INIT_BULK.super_cell_config))
            
    if len(config_list) == 0:
    # find relax config from relax dir
        config_list = search_files(relax_dir, "{}/*-{}".format(init_config_dirname, INIT_BULK.final_config))
        
    if len(config_list) == 0:
        # use init config
        config_list = [init_config_path]
    return config_list