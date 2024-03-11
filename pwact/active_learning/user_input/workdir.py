import os

"""
@Description :
parameter:
work_dir: first dir
type_dir: second dir

@Returns     :
@Author       :wuxingxing
"""

class WorkFile(object):
    """
    fixed dir construction :
    work_dir
    --------|train_type_dir
    ---------------------|log_dir
    ---------------------|model_dir
    such as ./work_dir/train_init_drop_10pct_dpnn means: 
            1. the model trained with 10% training data; 
            2. the newwork use dropout strategy;  
            3.the model type is dpnn
    """
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir
        self.tag_success = "tag.success"
        self.tag_error = "tag.error"

    def check_tag(self, prefix:str):
        if os.path.exists(os.path.join(prefix, self.tag_success)):
            return True
        else:
            return False

class TrainWork(WorkFile):
    def __init__(self, work_dir) -> None:
        super.__init__(work_dir)

    def set_model_init_infos(self, model_num: int=0, model_type: str="DP", init_data_path: str=None):
        self.model_num = model_num
        self.model_type = model_type
        self.init_data_path = init_data_path
    
    def set_train_datas(self, train_data_paths: list[str]):
        self.train_data_paths = train_data_paths

    def get_train_slurm_script(self, model_num_index:int =0):
        self.train_slurm_file = os.path.join("")

        self.log_dir = os.path.join(self.work_dir, "log_dir")
        self.model_dir = os.path.join(self.work_dir, "model_dir")
        self.train_kpu_dir = os.path.join(self.model_dir, "train_0_kpu_dir")
        self.md_kpu_dir = os.path.join(self.model_dir, "md_kpu_dir")

        self.config_yaml_path = os.path.join(self.work_dir, "train.yaml")
        self.md_kpu_config_yaml_path = os.path.join(self.work_dir, "md_kpu.yaml")
        self.md_kpu_script = os.path.join(self.work_dir, "md_kpu_slurm.job")
        self.train_kpu_script = os.path.join(self.work_dir, "train_kpu_slurm.job")
        self.train_script = os.path.join(self.work_dir, "train_slurm.job")
        self.train_success_tag = os.path.join(self.model_dir, "train_success.tag")
        self.md_kpu_success_tag = os.path.join(self.model_dir, "md_kpu_success.tag")
        self.train_kpu_success_tag = os.path.join(self.model_dir, "train_kpu_success.tag")

        self.model_save_path = os.path.join(self.model_dir, "checkpoint.pth.tar")

        # self.tensorbord_dir = os.path.join(self.work_dir, "tensorbord_dir")
        if os.path.exists(self.work_dir) is False:
            os.makedirs(self.work_dir)
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        if os.path.exists(self.model_dir) is False:
            os.makedirs(self.model_dir)
        
    
    """
    @Description :
    setting training data files path
    @Returns     :
    @Author       :wuxingxing
    """
    def set_torch_data_path(self, data_path = None):
        self.data_path = data_path
        self.train_data_path = []
        self.valid_data_path = []
        for i in range(len(self.data_path)):
            self.train_data_path.append(os.path.join(self.data_path[i], 'train_data/final_train'))
            self.valid_data_path.append(os.path.join(self.data_path[i], 'train_data/final_test'))
        
        print("work torch data paths are {}".format(self.data_path))

    def set_torch_data_sacle_infos(self, davg_dstd_dir):
        self.davg_dstd_dir = davg_dstd_dir
        
    def print_dir_tree(self):
        # latter would change by os.walk to prient the workdir tree
        print("the work dir is as listed:")
        print("work_dir:{}".format(self.work_dir))
        print("second_dir:{}\t{}\t{}\t".format(self.log_dir, self.model_dir, self.tensorbord_dir))
        print("set_torch_data_path:{}".format(self.data_path))

class WorkValidDir(object):
    """
    fixed dir construction :
    work_dir
    --------|train_type_dir
    ---------------------|log_dir
    ---------------------|model_dir
    ---------------------|tensorbord_record_dir
    such as ./work_dir/train_init_drop_10pct_dpnn means: 
            1. the model trained with 10% training data; 
            2. the newwork use dropout strategy;  
            3.the model type is dpnn
    """
    def __init__(self, work_path, model_path, p_path) -> None:
        self.work_dir = work_path
        self.model_dir = model_path
        self.p_dir = p_path
        self.data_path = None
        self.train_data_path = None
        self.valid_data_path = None

        if os.path.exists(self.work_dir) is False:
            os.makedirs(self.work_dir)

    """
    @Description :
    setting training data files path
    @Returns     :
    @Author       :wuxingxing
    """
    def set_torch_data_path(self, data_path = None):
        self.data_path = data_path
        self.train_data_path = []
        self.valid_data_path = []
        for i in range(len(self.data_path)):
            self.train_data_path.append(os.path.join(self.data_path[i], 'train_data/final_train'))
            self.valid_data_path.append(os.path.join(self.data_path[i], 'train_data/final_test'))
        
        print("work torch data paths are {}".format(self.data_path))

    def set_torch_data_sacle_infos(self, davg_dstd_dir):
        self.davg_dstd_dir = davg_dstd_dir
        

class WorkMDDir(object):
    def __init__(self, root_dir) -> None:
        self.work_dir = root_dir
        if os.path.exists(self.work_dir) is False:
            os.makedirs(self.work_dir)

        self.log_dir = os.path.join(self.work_dir, "log_dir")
        self.md_dir = os.path.join(self.work_dir, "md_dir")
        self.md_success_tag = os.path.join(self.md_dir, "md_success.tag")
        self.md_slurm_path = os.path.join(self.md_dir, "md_slurm.job")
        
        self.md_dpkf_dir = os.path.join(self.work_dir, "md_dpkf_dir") #the dir of pwmat+dpkf-movemenet to dpkf train for kpu analyse
        self.md_traj_dir = os.path.join(self.work_dir, "md_traj_dir")   #seperate movement to each atom_i.config and save to this path
        
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        if os.path.exists(self.md_dir) is False:
            os.makedirs(self.md_dir)
        if os.path.exists(self.md_dpkf_dir) is False:
            os.makedirs(self.md_dpkf_dir)
        if os.path.exists(self.md_traj_dir) is False:
            os.makedirs(self.md_traj_dir)
        
        self.set_md_files()

    def set_md_files(self):
        self.model_path = os.path.join(self.md_dir, "checkpoint.pth.tar")
        self.p_path = os.path.join(self.md_dir, "P.pt")
        self.atom_config_path = os.path.join(self.md_dir, "atom.config")
        self.fread_dfeat_path = os.path.join(self.md_dir, "fread_dfeat")
        self.feat_info_path = os.path.join(self.fread_dfeat_path, "feat.info")
        self.vdw_fitB_ntype_path = os.path.join(self.fread_dfeat_path, "vdw_fitB.ntype")
        self.md_input_path = os.path.join(self.md_dir, "md.input")
        self.gen_feat_path = os.path.join(self.md_dpkf_dir, "gen_feat.job")
        self.gen_feat_success_tag = os.path.join(self.md_dpkf_dir, "gen_feat_success.tag")

    def print_dir_tree(self):
        print("the work dir is as listed:")
        print("work_dir:{}".format(self.work_dir))
        print("second_dir:{}\t{}\t".format(self.log_dir, self.md_dir))

class WorkLabDir(object):
    def __init__(self, root_dir) -> None:
        self.work_dir = root_dir
        self.ab_dir = os.path.join(self.work_dir, "ab_dir")
        self.lab_dpkf_dir = os.path.join(self.work_dir, "lab_dpkf_dir")
        self.scf_slurm_path = os.path.join(self.ab_dir, "scf_slurm.job")
        self.scf_success_tag = os.path.join(self.ab_dir, "scf_success.tag")

        self.gen_feat_slurm_path = os.path.join(self.lab_dpkf_dir, "gen_feat_slurm.job")
        self.gen_feat_success_tag = os.path.join(self.lab_dpkf_dir, "gen_feat_success.tag")

        if os.path.exists(self.work_dir) is False:
            os.makedirs(self.work_dir)
        if os.path.exists(self.ab_dir) is False:
            os.makedirs(self.ab_dir)
        if os.path.exists(self.lab_dpkf_dir) is False:
            os.makedirs(self.lab_dpkf_dir)

if __name__=="__main__":
    print()