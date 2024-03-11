import os

from pwact.utils.file_operation import search_files, link_file, copy_file, write_to_file
from pwact.utils.constant import INIT_BULK, DFT_STYLE, TEMP_STRUCTURE

from pwact.active_learning.user_input.resource import Resource
from pwact.active_learning.user_input.init_bulk_input import InitBulkParam

from pwact.data_format.configop import do_super_cell, do_scale, do_pertub

def  duplicate_scale(resource: Resource, input_param:InitBulkParam):
    #work dir
    init_configs = input_param.sys_config
    relax_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
                 
    for init_config in init_configs:
        # print(init_config.super_cell)
        config_format = init_config.format#
        config_file = init_config.config_file
        init_config_name = "init_config_{}".format(init_config.config_index)
        if init_config.relax:
            config_file = os.path.join(relax_dir, init_config_name, DFT_STYLE.get_relaxed_config(resource.dft_style))
            config_format = DFT_STYLE.get_format_by_postfix(DFT_STYLE.get_relaxed_config(resource.dft_style))
        else:
            config_file = init_config.config_file
        super_cell_scale_dir = os.path.join(duplicate_scale_dir, init_config_name)
        super_cell_config = None
        if init_config.super_cell is not None:
            if not os.path.exists(super_cell_scale_dir):
                os.makedirs(super_cell_scale_dir)
            # super_content, lattic_index = super_cell(init_config.super_cell, config, super_cell_config)

            super_cell_config = os.path.join(super_cell_scale_dir, DFT_STYLE.get_super_cell_config(resource.dft_style))

            if not os.path.exists(super_cell_config):
                do_super_cell(config=config_file,
                    input_format=config_format,
                    supercell_matrix=init_config.super_cell, 
                    pbc=init_config.pbc, 
                    direct = True, 
                    sort = True,
                    save_format=DFT_STYLE.get_pwdata_format(resource.dft_style), #cp2k ->vasp/poscar
                    save_path=super_cell_scale_dir, 
                    save_name=DFT_STYLE.get_super_cell_config(resource.dft_style))
            config_format = DFT_STYLE.get_pwdata_format(resource.dft_style)
            
            
        if init_config.scale is not None:
            for s_index, scale in enumerate(init_config.scale):
                # save_scale_config = os.path.join(super_cell_scale_dir, "{}_{}".format(scale, DFT_STYLE.get_scale_config(init_config.dft_style)))
                if not os.path.exists(super_cell_scale_dir):
                    os.makedirs(super_cell_scale_dir)
                if super_cell_config is not None:
                    input_scale_config = super_cell_config # config from super cell
                else:
                    input_scale_config = config_file # config from relax or init config
                do_scale(config=input_scale_config,
                    input_format = config_format,
                    scale_factor=scale, 
                    direct=True,
                    sort=True, 
                    save_format=DFT_STYLE.get_pwdata_format(resource.dft_style),
                    save_path=super_cell_scale_dir, 
                    save_name="{}_{}".format(scale, DFT_STYLE.get_scale_config(resource.dft_style))
                    )

def do_post_duplicate_scale(resource: Resource, input_param:InitBulkParam):
    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
    real_duplicate_scale_dir = os.path.join(input_param.root_dir, INIT_BULK.super_cell_scale) 
    if os.path.exists(duplicate_scale_dir):
        link_file(duplicate_scale_dir, real_duplicate_scale_dir)
    tag = os.path.join(duplicate_scale_dir, INIT_BULK.tag_super_cell)
    if not os.path.exists(tag):
        write_to_file(tag, "super cell and scale done ", "w")

def duplicate_scale_done(input_param:InitBulkParam):
    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
    tag = os.path.join(duplicate_scale_dir, INIT_BULK.tag_super_cell)
    return True if os.path.exists(tag) else False

'''
description: 
param {str} work_dir the dir of atom.config file
param {int} pert_num
param {float} cell_pert_fraction
param {float} atom_pert_distance
return {*}
author: wuxingxing
'''
def do_pertub_work(resource: Resource, input_param:InitBulkParam):
    #work dir 
    # pert_num:int=50
    # cell_pert_fraction:float=0.03
    # atom_pert_distance:float=0.01
    init_configs = input_param.sys_config
    relax_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.relax)
    duplicate_scale_dir = os.path.join(input_param.root_dir, TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.super_cell_scale) 
    pertub_dir = os.path.join(input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
    # real_pertub_dir = os.path.join(input_param.root_dir,INIT_BULK.pertub)
    # make pertub dirs
    for init_config in init_configs:
        if init_config.perturb is None:
            continue
        init_config_name = "init_config_{}".format(init_config.config_index)
        pert_config_list, config_type = get_config_files_with_order(
            super_cell_scale_dir=duplicate_scale_dir,
            relax_dir=relax_dir,
            init_config_dirname=init_config_name, 
            init_config_path=init_config.config_file, 
            pertub_dir=None,
            dft_style=resource.dft_style
            )
        print(pert_config_list)
        for index, config in enumerate(pert_config_list):
            if config_type == INIT_BULK.scale: 
                tmp_config_dir = os.path.basename(os.path.basename(config).replace(DFT_STYLE.get_postfix(resource.dft_style), "")) # 0.8_scale.poscar or 0.8_scale.config -> 0.8_scale
            else:
                tmp_config_dir = config_type
            work_dir = os.path.join(pertub_dir, init_config_name, tmp_config_dir)
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            target_config = os.path.join(work_dir, os.path.basename(config))
            copy_file(config, target_config)
            
            do_pertub(config=target_config, 
                input_format=DFT_STYLE.get_format_by_postfix(os.path.basename(target_config)),
                pert_num=init_config.perturb, 
                cell_pert_fraction=init_config.cell_pert_fraction, 
                atom_pert_distance=init_config.atom_pert_distance, 
                direct=True,
                sort=True, 
                save_format=DFT_STYLE.get_pwdata_format(resource.dft_style), 
                save_path=work_dir, 
                save_name=DFT_STYLE.get_pertub_config(resource.dft_style)
                )

def do_post_pertub(resource: Resource, input_param:InitBulkParam):
    pertub_dir = os.path.join(input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
    real_pertub_dir = os.path.join(input_param.root_dir,INIT_BULK.pertub)
    if os.path.exists(pertub_dir):
        link_file(pertub_dir, real_pertub_dir)
    tag = os.path.join(pertub_dir, INIT_BULK.tag_pertub)
    if not os.path.exists(tag):
        write_to_file(tag, "pertub done ", "w")

def pertub_done(input_param:InitBulkParam):
    pertub_dir = os.path.join(input_param.root_dir,TEMP_STRUCTURE.tmp_init_bulk_dir, INIT_BULK.pertub)
    tag = os.path.join(pertub_dir, INIT_BULK.tag_pertub)
    return True if os.path.exists(tag) else False

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
def get_config_files_with_order(super_cell_scale_dir:str, relax_dir:str, init_config_dirname:str, init_config_path:str, pertub_dir:str=None, dft_style:str=None):
    config_list = []
    config_type = INIT_BULK.init
    if pertub_dir is not None:
        config_list = search_files(pertub_dir, "{}/*/*{}".format(init_config_dirname, DFT_STYLE.get_pertub_config(dft_style)))
        config_type = INIT_BULK.pertub
    if len(config_list) == 0:
    # find scaled configs under super_cell_scale dir
        config_list = search_files(super_cell_scale_dir, "{}/*{}".format(init_config_dirname, DFT_STYLE.get_scale_config(dft_style)))
        config_type = INIT_BULK.scale
    if len(config_list) == 0:
    # find super cell config under super_cell_scale dir
        config_list = search_files(super_cell_scale_dir, "{}/{}".format(init_config_dirname, DFT_STYLE.get_super_cell_config(dft_style)))
        config_type = INIT_BULK.super_cell

    if len(config_list) == 0:
    # find relax config from relax dir
        config_list = search_files(relax_dir, "{}/{}".format(init_config_dirname, DFT_STYLE.get_relaxed_config(dft_style)))
        config_type = INIT_BULK.relax

    if len(config_list) == 0:
        # use init config
        config_list = [init_config_path]
        config_type = INIT_BULK.init

    return config_list, config_type