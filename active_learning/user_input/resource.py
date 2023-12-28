from utils.file_operation import get_required_parameter, get_parameter

class Resource(object):
    # _instance = None

    def __init__(self, json_dict:dict) -> None:
        self.train_resource = self.get_resource(get_required_parameter("train", json_dict))
        self.explore_resource = self.get_resource(get_required_parameter("explore", json_dict))
        self.scf_resource = self.get_resource(get_required_parameter("scf", json_dict))

    # @classmethod
    # def get_instance(cls, json_dict:dict = None):
    #     if not cls._instance:
    #         cls._instance = cls(json_dict)
    #     return cls._instance
    
    def get_resource(self, json_dict:dict):
        group_size = get_parameter("group_size", json_dict, None)
        number_node = get_required_parameter("number_node", json_dict)
        gpu_per_node = get_parameter("gpu_per_node", json_dict, None)
        cpu_per_node = get_parameter("cpu_per_node", json_dict, None)
        queue_name = get_required_parameter("queue_name", json_dict)
        custom_flags = get_parameter("custom_flags", json_dict, [])
        source_list = get_parameter("source_list", json_dict, [])
        module_list = get_parameter("module_list", json_dict, [])
        resource = ResourceDetail(group_size, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, source_list, module_list)
        return resource

class ResourceDetail(object):
    def __init__(self, group_size:int , number_node:int , gpu_per_node:int , cpu_per_node:int ,\
                  queue_name:str, custom_flags:list[str], source_list:list[str], module_list:list[str]) -> None:
        self.group_size = group_size
        self.number_node = number_node
        self.gpu_per_node = gpu_per_node
        self.cpu_per_node = cpu_per_node
        self.queue_name = queue_name
        self.custom_flags = custom_flags
        self.source_list = source_list
        self.module_list = module_list

        if self.gpu_per_node is None and self.cpu_per_node is None:
            raise Exception("ERROR! Both CPU and GPU resources are not specified!")
        

    def set_source_list(self):
        pass

    def set_module_list(self):
        pass

    def set_custom_flag(self):
        pass

# class ResourceExplore(ResourceTrain):
#     def __init__(self, group_size, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, source_list, module_list) -> None:
#         super.__init__(group_size, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, source_list, module_list)

# class ResourceSCF(ResourceTrain):
#     def __init__(self, group_size, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, source_list, module_list) -> None:
#         super.__init__(group_size, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, source_list, module_list)
