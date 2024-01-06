import os
import json
from utils.file_operation import write_to_file
from utils.pre_al_data_util import get_movement_num

from utils.movement2traindata import Scf2Movement
from utils.movement2traindata import movement2traindata

import shutil

class AbGenerator(object):
    def __init__(self, itername, source_atom_path, upf_list, md_config, work_dir):
        self.itername = itername
        self.source_atom_path = source_atom_path
        self.work_dir = work_dir
        self.upf_list = upf_list
        self.md_config = md_config
        self.success_tag = os.path.join(self.work_dir, "ab_init_success_tag")
        self.pre_precess()
        # self.do_labeling()

    def make_etot_input(self, save_path, upf_list, job_type = 'md', ecut=50.0):
        # 4 1
        # in.psp1 = Ga.SG15.PBE.UPF
        # in.psp2 = As.SG15.PBE.UPF
        # job = scf
        # in.atom = atom.config
        # out.force = T
        # ecut = 50.0
        res = ""
        res += "4 1\n"
        for i in range(len(upf_list)):
            res += "in.psp{} = {}\n".format(i+1, upf_list[i])
        res += "job = {}\n".format(job_type)
        res += "in.atom = atom.config\n"
        res += "energy_decomp = T\n"
        res += "ecut = {}\n".format(ecut)
        if job_type == 'md':
            res += "md_detail = 1 {} {} {} {} \n".format(self.md_config["MD_steps"], self.md_config["step_time"], self.md_config["temp_start"], self.md_config["temp_end"])
        with open(save_path, "w") as wf:
            wf.write(res)

    def pre_precess(self):
        if os.path.exists(self.work_dir) is False:
            os.makedirs(self.work_dir)
        # copy atom.config
        atom_config = os.path.join(self.work_dir, "atom.config")
        if os.path.exists(atom_config) is False:
            shutil.copyfile(self.source_atom_path, atom_config)
        # ln UPF files
        upfs = []
        for upf in self.upf_list:
            basename = os.path.basename(upf)
            if os.path.exists(os.path.join(self.work_dir, basename)) is False:
                shutil.copyfile(upf, os.path.join(self.work_dir, basename))#test if need filename
            upfs.append(basename)
        self.make_etot_input(os.path.join(self.work_dir, "etot.input"), upfs)

    def do_labeling(self):
            if os.path.exists(self.success_tag):
                print("{} already be done!".format(self.work_dir))
                return
            
            cwd = os.getcwd()
            path_list = os.listdir(self.work_dir.ab_dir)
            for i in path_list:
                atom_config_path = os.path.join(self.work_dir.ab_dir, "{}/etot.input".format(i))
                if os.path.exists(atom_config_path):
                    if os.path.exists(os.path.join(self.work_dir.ab_dir, "REPORT")) is False:
                        os.chdir(os.path.dirname(atom_config_path))
                        commands = "mpirun -np 4 PWmat"
                        res = os.system(commands)
                        if res != 0:
                            raise Exception("run md command {} error!".format(commands))
                        os.chdir(cwd)
                        # construct the atom.config to MOVEMENT by using REPORT, OUT.FORCE
                        Scf2Movement(atom_config_path, \
                            os.path.join(os.path.join(self.work_dir.ab_dir, "OUT.FORCE")), \
                            os.path.join(os.path.join(self.work_dir.ab_dir, "REPORT")), \
                            os.path.join(os.path.join(self.work_dir.ab_dir, "MOVEMENT")),)
                    print("{} labeling is done! ".format(atom_config_path))
            write_to_file(self.work_dir.label_success_tag, '1')

            self.post_precess()
            
    """
    @Description :
    set labeling result to iter_result.json
    if cur iter num of images + pre images < 10, just save base info
    else:
        ln pre images to cur pwdata dir and convert to dpkf data
    @Returns     :
    @Author       :wuxingxing
    """
    def post_precess(self):
        iter_result_json_path = os.path.join(os.getcwd(), "iter_result.json")
        iter_result_json = json.load(open(iter_result_json_path)) if os.path.exists(iter_result_json_path) else {}

        path_list = os.listdir(self.work_dir.ab_dir)
        MOVEMENT_list = []
        for i in path_list:
            MOVEMENT_path = os.path.join(self.work_dir.ab_dir, "{}/MOVEMENT".format(i))
            if os.path.exists(MOVEMENT_path):
                target_save_dir = os.path.join(self.work_dir.lab_dpkf_dir, "PWdata/{}".format(i))
                if os.path.exists(target_save_dir) is False:
                    os.makedirs(target_save_dir)
                target_movement = os.path.join(target_save_dir, "MOVEMENT")
                if os.path.exists(target_movement) is False:
                    os.symlink(MOVEMENT_path, target_movement)
                MOVEMENT_list.append(os.path.abspath(target_movement))
        
        iter_result = {}
        iter_result["cur_MOVEMENT_num"] = len(MOVEMENT_list)
        iter_result["cur_MOVEMENT_paths"] = MOVEMENT_list

        # if new movements accumulated than 10, convert them to dpkf data
        older_movements = self.get_movement_num(self.itername, iter_result_json)
        if len(MOVEMENT_list) + len(older_movements) >= 10:
            
        # ln older movements to cur movement dir
            for i in older_movements:
                base_dir = os.path.dirname(i)
                base_dir_name = os.path.basename(base_dir)
                target_path = os.path.join(self.work_dir.lab_dpkf_dir, base_dir_name)
                if os.path.exists(target_path) is False:
                    os.symlink(base_dir, target_path)
                    
            parameter_path = self.MD_info["parameter_path"]
            dpkf_data = os.path.join(self.work_dir.lab_dpkf_dir, "PWdata")
            movement2traindata(dpkf_data, parameter_path)
            iter_result["dpkf_data"] = dpkf_data

        iter_result_json[self.itername] = iter_result

        json.dump(iter_result_json, open(iter_result_json_path, "w"))

if __name__ == "__main__":
    atom_config_dir = "/home/wuxingxing/datas/system_config/GaAs_system/atom_configs/init"
    atom_config_list = [5, 10, 15, 20, 25, 30]
    save_path = "/home/wuxingxing/datas/system_config/GaAs_system/ab_md"
    upf_list = ["/home/wuxingxing/datas/system_config/GaAs_system/As.SG15.PBE.UPF", "/home/wuxingxing/datas/system_config/GaAs_system/Ga.SG15.PBE.UPF"]
    md_config = {"MD_steps":1000, "step_time":1, "temp_start":300, "temp_end":300}
     
    for i in atom_config_list:
        iter_name = "ab_{}".format(i)
        source_atom_path = os.path.join(atom_config_dir, "atom_{}.config".format(i))
        work_dir = os.path.join(save_path, "ab_{}".format(i))
        AbGenerator(iter_name, source_atom_path, upf_list, md_config, work_dir)
