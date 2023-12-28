import os
import shutil
import re
import json

from utils.constant import SCF_FILE_STRUCTUR
from utils.format_input_output import make_iter_name, make_train_name

'''
description: 
save json_dict to save_path, if file dir is not exist, create it.
param {dict} json_dict
param {*} save_path
return {*}
author: wuxingxing
'''
def save_json_file(json_dict:dict, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    json.dump(json_dict, open(save_path, "w"), indent=4)

"""
@Description :
读取最后一行文件
@Returns     :
@Author       :wuxingxing
"""
def file_read_last_line(file_path, type_name="int"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            last_line = rf.readlines()[-1]  #the last line
            if '[]' in last_line:
                return []
            last_line = last_line.replace(" ","")[1:-2].split(',')
    if len(last_line) > 0 and type_name == "int":
        last_line = [int(i) for i in last_line]
    if len(last_line) > 0 and type_name == "float":
        last_line = [float(i) for i in last_line]
    return last_line

def file_read_lines(file_path, type_name="float"):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as rf:
            lines = rf.readlines()  #the last line
            for line in lines:
                line = re.sub('[\[\]\\n]','',line)
                if len(line) > 0 and type_name == "int":
                    line = [int(i) for i in line.split(',')]
                if len(line) > 0 and type_name == "float":
                    line = [float(i) for i in line.split(',')]
                data.append(line)
    return data

'''
description: 
 save line str to file_path
param {*} file_path
param {*} line
param {*} mode :'a' or 'w', default is 'a'
return {*}
author: wuxingxing
'''
def write_to_file(file_path, line, mode='a'):
    with open(file_path, mode) as wf:
        wf.write(line)

"""
@Description :
删除指定目录下所有文件(该目录不删除) / 或者删除指定文件名文件
@Returns     :
@Author       :wuxingxing
"""

def del_file(path_dir):
    if os.path.exists(path_dir) == False:
        return

    if os.path.isfile(path_dir):
        os.remove(path_dir)
        return
        
    for i in os.listdir(path_dir) :
        file_path = os.path.join(path_dir, i)
        if os.path.isfile(file_path) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_path)
        else:
            del_file(file_path)

'''
description: 
    mv files under target_dir to source_dir
param {str} source_dir
param {str} target_dir
return {*}
author: wuxingxing
'''
def mv_dir(source_dir:str, target_dir:str):
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    files = os.listdir(source_dir)
    for file in files:
        shutil.move(os.path.join(source_dir, file), target_dir)
    del_dir(source_dir)

'''
description: 
    delete dir or file if exists.
param {str} del_dir
return {*}
author: wuxingxing
'''
def del_dir(del_dir:str):
    if os.path.exists(del_dir):
        shutil.rmtree(del_dir)
        
'''
description: 
    get value of param in json_input which is required parameters which need input by user
        if the parameter is not specified in json_input, raise error and print error log to user.
param {str} param
param {dict} json_input
param {str} info 
return {*}
author: wuxingxing
'''
def get_required_parameter(param:str, json_input:dict):
    if param not in json_input.keys():
        raise Exception("Input error! : The {} parameter is missing and must be specified in input json file!".format(param))
    return json_input[param]

'''
description: 
    get value of param in json_input,
        if the parameter is not specified in json_input, return the default parameter value 'default_value'
param {str} param
param {dict} json_input
param {*} default_value
return {*}
author: wuxingxing
'''
def get_parameter(param:str, json_input:dict, default_value, format_type:str=None):
    res = None
    if param not in json_input.keys():
        res = default_value
    else:
        res = json_input[param]
    
    if format_type is not None:
        if format_type.upper() == "upper".upper():
            res = res.upper()
    return res

'''
description: 
    search mvm files from iter.0000 to iter.current
param {str} root_dir
param {str} current_itername
return {*}
author: wuxingxing
'''
def search_mvm_files(root_dir:str, current_itername:str):
    mvm_list = []
    current_iter = int(current_itername.split(".")[1])
    for iter in range(0, current_iter):
        iter_name = make_iter_name()
        iter_scf_dir = os.path.join(root_dir, iter_name, SCF_FILE_STRUCTUR.RESULT)
    print("not realized yet!")
    return mvm_list

