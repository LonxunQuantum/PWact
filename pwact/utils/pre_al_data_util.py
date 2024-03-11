"""
this script inculding the functions for data coverting in active learning iters
"""

from pwact.utils.format_input_output import make_iter_name
import os

"""
@Description :
    this function should not be used, when the new labeled data nums are less, just ignore them
    return data paths
@Returns     :
@Author       :wuxingxing
"""
def get_movement_num(iter_result):
    keys = list(iter_result.keys()) #反向
    keys = sorted(keys, key=lambda x:int(x[5:]), reverse=True)

    movement_list = []
    movement_from_iter = []
    for i in keys:
        if "feature_path" not in iter_result[i].keys() and len(iter_result[i]["movement_file"]) > 0:
            movement_list.extend([os.path.join(iter_result[i]["movement_dir"], _) for _ in iter_result[i]["movement_file"]])
            movement_from_iter.append(i)
    
    return movement_list, movement_from_iter

"""
@Description :
get training data path
if first training: return init_data
else: return all feature_path

the init data dir should be set to the last path, when the new scale data info comes,
they will be saved to the init data path.

data_limit: when new datas more than this value, retrain the model.
@Returns     :
@Author       :wuxingxing
"""
def get_feature_data_path(self):
    retain_sign = False
    iter_index = int(self.itername[5:])
    if iter_index == 0:
        return self.system_info["init_data_path"], True

    pre_iter_name = make_iter_name(iter_index-1)
    if "feature_path" not in self.iter_result[pre_iter_name].keys():
        return [], False

    data_paths = set()
    keys = list(self.iter_result.keys()) #反向
    keys = sorted(keys, key=lambda x:int(x[5:]), reverse=True)
    for i in keys:
        if "feature_path" in self.iter_result[i].keys():
            data_paths.add(self.iter_result[i]["feature_path"])
            retain_sign = True
    data_paths = list(data_paths)
    data_paths.extend(self.system_info["init_data_path"])
    # iter_res_path = "{}/iter_result.json".format(system_info["work_root_path"])
    # json.dump(iter_result, open(iter_res_path, "w"), indent=4)
    return data_paths, retain_sign

'''
Description: 
 get dir name of images which under the 'path'
param {*} path
param {*} basename
Returns: 
    ['image_000', 'image_001', ..., 'image_N']
Author: WU Xingxing
'''
def get_image_nums(path, basename="image"):
    dirlist = os.listdir(path)
    res = []
    for d in dirlist:
        if basename in d:
            res.append(d)
    res = sorted(res, key=lambda x: int(x.split('_')[-1]))
    return res
