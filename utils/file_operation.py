import os
import re

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

"""
@Description :
存储行 array 数据到txt最后一行
@Returns     :
@Author       :wuxingxing
"""
def write_to_file(file_path, line):
    with open(file_path, 'a') as wf:
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
def get_parameter(param:str, json_input:dict, default_value):
    if param not in json_input.keys():
        return default_value
    else:
        return json_input[param]

