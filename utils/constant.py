from enum import Enum

class MDTYPE:
    fortran_lmps = 1 #
    libtorch_lmps = 2 # 
    main_md = 3 #

class AL_STRUCTURE:
    train = "train"
    explore = "explore"
    labeling = "label"
    
class TRAIN_FILE_STRUCTUR:
    work_dir = "work_dir"
    feature_dir = "feature"
    feature_json = "feature.json"
    feature_job = "feature.job"
    feature_tag = "tag.feature.success"
    train_json = "train.json"
    train_job = "train.job"
    train_tag = "tag.train.success"
    movement = "MOVEMENT"

class MODEL_CMD:
    train = "train"
    gen_feat = "gen_feat"
    test = "test"
    
class SCF_FILE_STRUCTUR:
    RESULT = ""

class EXPLORE_FILE_STRUCTURE:
    RESULT = ""

class TRAIN_INPUT_PARAM:
    train_mvm_files = "train_movement_file"
    train_feature_path = "train_feature_path"
    reserve_feature = "reserve_feature" #False
    reserve_work_dir = "reserve_work_dir" #False
    valid_shuffle = "valid_shuffle" #True
    train_valid_ratio = "train_valid_ratio" #0.8
    seed = "seed" #2023
    recover_train = "recover_train" #true


class ENSEMBLE:
    npt_tri = "npt_tri",
    nvt = "nvt"
    