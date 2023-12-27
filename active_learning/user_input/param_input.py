from utils.file_operation import get_required_parameter, get_parameter

class ParamInput(object):
    def __init__(self, json_dict: dict) -> None:
        self.work_dir = get_parameter("work_dir", json_dict, "work_dir")
        self.init_mvm_files = get_parameter("init_mvm_file", json_dict, [])
        
        self.train_param = TrainParam(json_dict)

class TrainParam(object):
    def __init__(self, json_dict) -> None:
        self.train_input_file = get_parameter("work_dir", json_dict, None)
        if self.train_input_file is not None:
            
        else:
            

class ExploreParam(object):
    def __init__(self, json_dict) -> None:
        pass

class SCFParam(object):
    def __init__(self) -> None:
        pass