import os
from typing import Optional


class Job(object):
    def __init__(
            self,
            partition:Optional[str],
            name:Optional[str],
            num_nodes:Optional[int],
            num_tasks_per_node:Optional[int],
            threads_per_core:Optional[int],
            num_gpus:Optional[int],
            num_gpus_per_task:Optional[int],
            ):
        '''
        Description
        -----------
            1. 定义一个slurm任务
        
        
        Parameters
        ----------
            1. partition: str
                - slurm 的队列/任务分区
            2. name: str
                - 任务的名字
            3. num_nodes: int
                - 分配的节点数目
            4. num_tasks_per_node: int
                - 每个node的任务数目
            5. threads_per_core: int
                - 每个core的最大thread数目
            6. num_gpus: int
                - gpu 的数目
            7. num_gpus_per_task: int
                - 每个 task 分配的 GPU 的数目
        
        Note
        ----    
            1. 一个 node 可以分配多个 task
            2. 一个 task 可以分配多个 gpu
        '''
        self.partition = partition
        self.name = name
        self.num_nodes = num_nodes
        self.num_tasks_per_node = num_tasks_per_node
        self.num_threads_per_core = threads_per_core
        self.num_gpus = num_gpus
        self.num_gpus_per_task = num_gpus_per_task
        self.command = self.attach_command()
    
    
    def attach_command(self):
        return None
    
    
    def to(self, filename:str):
        current_path = os.getcwd()
        with open(os.path.join(current_path, filename), 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --partition={0}\n".format(self.partition))
            f.write("#SBATCH --job-name={0}\n".format(self.name))
            f.write("#SBATCH --nodes={0}\n".format(self.num_nodes))
            f.write("#SBATCH --ntasks-per-node={0}\n".format(self.num_tasks_per_node))
            f.write("#SBATCH --threads-per-core={0}".format(self.num_threads_per_core))
            f.write("#SBATCH --gres=gpu:{0}".format(self.num_gpus))
            f.write("#SBATCH --gpus-per-task={0}".format(self.num_gpus_per_task))
            f.write("\n\n\n")
            f.write("{0}".format(self.command))