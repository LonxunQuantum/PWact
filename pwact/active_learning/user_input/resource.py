from pwact.utils.json_operation import get_parameter, get_required_parameter
from pwact.utils.constant import AL_WORK, DFT_STYLE, SLURM_OUT, CP2K, LAMMPS

class Resource(object):
    # _instance = None
    def __init__(self, json_dict:dict, job_type:str=AL_WORK.run_iter, dft_style:str=None) -> None:
        if job_type == AL_WORK.run_iter:
            self.train_resource = self.get_resource(get_required_parameter("train", json_dict))
            if self.train_resource.number_node > 1:
                self.train_resource.number_node = 1
            if self.train_resource.gpu_per_node > 1:
                self.train_resource.gpu_per_node = 1
            if self.train_resource.cpu_per_node > 1:
                self.train_resource.cpu_per_node = 1
            print("Warining: the resouce of node, gpu per node and cpu per node  in training automatically adjust to [1, 1, 1]")
            self.train_resource.command = self.train_resource.command.upper()
             
            self.explore_resource = self.get_resource(get_required_parameter("explore", json_dict))
            self.explore_resource.command = "{} -in {} > {}".format(self.explore_resource.command, LAMMPS.input_lammps, SLURM_OUT.md_out)

        # check dft resource
        self.dft_resource = self.get_resource(get_required_parameter("dft", json_dict))
        
        # dftb_command = get_parameter("dftb_command", json_dict["dft"], None)
        # if dftb_command is not None:
        #     self.dft_resource.dftb_command  = "{} > {}".format(dftb_command, SLURM_OUT.dft_out)
        self.dft_style = dft_style
        if DFT_STYLE.vasp.lower() == dft_style.lower():
            self.dft_resource.command = "{} > {}".format(self.dft_resource.command, SLURM_OUT.dft_out)
        elif DFT_STYLE.pwmat.lower() == dft_style.lower():
            self.dft_resource.command = "{} > {}".format(self.dft_resource.command, SLURM_OUT.dft_out)
        elif DFT_STYLE.cp2k.lower() == dft_style.lower():
            self.dft_resource.command = "{} {} > {}".format(self.dft_resource.command, CP2K.cp2k_inp, SLURM_OUT.dft_out)

    # @classmethod
    # def get_instance(cls, json_dict:dict = None):
    #     if not cls._instance:
    #         cls._instance = cls(json_dict)
    #     return cls._instance
    
    def get_resource(self, json_dict:dict):
        command = get_required_parameter("command", json_dict)
        group_size = get_parameter("group_size", json_dict, 1)
        parallel_num = get_parameter("parallel_num", json_dict, 1)
        number_node = get_parameter("number_node", json_dict, 1)
        gpu_per_node = get_parameter("gpu_per_node", json_dict, 0)
        cpu_per_node = get_parameter("cpu_per_node", json_dict, 1)
        queue_name = get_required_parameter("queue_name", json_dict)
        queue_name = queue_name.replace(" ","")
        custom_flags = get_parameter("custom_flags", json_dict, [])
        source_list = get_parameter("source_list", json_dict, [])
        module_list = get_parameter("module_list", json_dict, [])
        env_list = get_parameter("env_list", json_dict, [])

        env_script = ""
        if len(source_list) > 0:
            for source in source_list:
                if "source" != source.split()[0].lower():
                    tmp_source = "source {}\n".format(source)
                else:
                    tmp_source = "{}\n".format(source)
                env_script += tmp_source
        
        if len(env_list) > 0:
            for source in env_list:
                env_script += source + "\n"

        if len(module_list) > 0:
            for source in module_list:
                if "module" != source.split()[0].lower():
                    tmp_source = "module load {}\n".format(source)
                else:
                    tmp_source = "{}\n".format(source)
                env_script += tmp_source

        resource = ResourceDetail(command, group_size, parallel_num, number_node, gpu_per_node, cpu_per_node, queue_name, custom_flags, env_script)
        return resource

class ResourceDetail(object):
    def __init__(self, command:str, group_size:int , parallel_num:int, number_node:int , gpu_per_node:int , cpu_per_node:int ,\
                  queue_name:str, custom_flags:list[str], env_script:str) -> None:
        self.command = command
        self.group_size = group_size
        self.parallel_num = parallel_num
        self.number_node = number_node
        self.gpu_per_node = gpu_per_node
        self.cpu_per_node = cpu_per_node
        self.queue_name = queue_name
        self.custom_flags = custom_flags
        self.env_script = env_script

        if self.gpu_per_node is None and self.cpu_per_node is None:
            raise Exception("ERROR! Both CPU and GPU resources are not specified!")