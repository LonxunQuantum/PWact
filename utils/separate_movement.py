# -*- coding: utf-8 -*-

"""
    Image shuffling    
"""
import numpy as np 
import os
import math

class MovementOp:

    def __init__(self, MOVEMENT_path):

        self.MOVEMENT_path = MOVEMENT_path
        self.all_image, self.num_image, self.atom_nums = self.get_all_image()
    
    """this bug fixed by wlj, this function does not need"""
    def add_Atomic_Energy_block(self, save_path):
        self.add_Atomic_Energy_line()
        self.save_all_image_as_one_movement(save_path)

    def add_Atomic_Energy_line(self):
        def get_itype_list(image):
            start = False
            itypes = []
            for i in image:
                if "Force (-force," in i:
                    break
                if "Position (normalized)" in i:
                    start = True
                elif start:
                    itypes.append(int(i.split()[0]))
            return itypes

        itypes = get_itype_list(self.all_image[0])
        if len(itypes) != self.atom_nums:
            raise Exception
        for i in range(len(self.all_image)):
            res = "Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  -0.2028517919E+06\n"
            for j in self.all_image[i]:
                if "Etot,Ep,Ek (eV)" in j:
                    ep = float(j.split("=")[2].strip().split( )[1])/self.atom_nums
                    break
            last_line = self.all_image[i].pop()
            self.all_image[i].append(res)
            for t in itypes:
                self.all_image[i].append("  {}   {}   {}    {}\n".format(t, ep, 0.0, 0.0))
            self.all_image[i].append(last_line)

    def separate_movement_2part_by_ratio(self, save_path, ratio):
        start = 0
        mid = self.num_image - math.floor(self.num_image * ratio)
        end = self.num_image
        movement = os.path.join(save_path, "train_MOVEMENT")
        with open(movement, 'w') as wf:
            for i in range(start, mid):
                for j in self.all_image[i]:
                    wf.write(j)
        print("{} saved.".format(movement))
        
        movement = os.path.join(save_path, "valid_MOVEMENT")
        with open(movement, 'w') as wf:
            for i in range(mid, end):
                for j in self.all_image[i]:
                    wf.write(j)
        print("{} saved.".format(movement))

        print("end info: image{}-{} save to MOVEMENT_train; image{}-{} save to MOVEMENT_valid".format(start, mid-1, mid, end))

    '''
    Description:
    Take a cut of md from MOVEMENT 
    param {*} self
    param {*} save_path
    param {*} ratio
    Returns: 
    Author: WU Xingxing
    '''
    def cut_movement_by_indexs(self, save_path, start, end):
        movement = os.path.join(save_path, "MOVEMENT")
        with open(movement, 'w') as wf:
            for i in range(start, end):
                for j in self.all_image[i]:
                    wf.write(j)
        print("{} saved.".format(movement))
        print("end info: image{}-{} save to MOVEMENT".format(start, end))


    """
    @Description :
        save each images as atom.config
    @Returns     :
    @Author       :wuxingxing
    """
    def save_each_image_as_atom_config(self, atom_config_save_dir, interval=1):
        if os.path.exists(atom_config_save_dir) is False:
            os.makedirs(atom_config_save_dir)
        for i in range(len(self.all_image)):
            if i % interval == 0:
                atom_config = os.path.join(atom_config_save_dir, "atom_{}.config".format(i))
                with open(atom_config, 'w') as wf:
                    for j in self.all_image[i]:
                        wf.write(j)
                print("{} saved.".format(atom_config))

    '''
    description: 
        sample images from movement after each interval. 
    param {*} self
    param {*} movement_save_path
    param {*} interval
    return {*}
    author: wuxingxing
    Description: 
    Returns: 
    Author: WU Xingxing
    '''
    def save_image_by_interval(self, movement_save_path, interval=1):
        for i in range(len(self.all_image)):
            if i % interval == 0:
                with open(movement_save_path, 'w') as wf:
                    for j in self.all_image[i]:
                            wf.write(j)
        print("{} saved.".format(movement_save_path))

    """
    @Description :
        save image by its index in movement
    @Returns     :
    @Author       :wuxingxing
    """
    def save_image_as_atom_config(self, file_save_path, index):
        atom_config = file_save_path
        with open(atom_config, 'w') as wf:
            for j in self.all_image[index]:
                wf.write(j)
        print("{} saved.".format(atom_config))

    """
    @Description :
        save the images by it's index, for example:
         a movement with 1000 images, when the start is 200, and end is 400, we will save the image[200:400] to a movement file.
    @Returns     :
    @Author       :wuxingxing
    """
    def save_images_by_index(self, movement_save_path, start=0, end = -1, patten="w"):
        with open(movement_save_path, patten) as wf:
            for j in self.all_image[start:end]:
                for i in j:
                    wf.write(i)
        print("{} saved.".format(movement_save_path))

    """
    @Description :
        extract last md of the movement as atom.config
    @Returns     :
    @Author       :wuxingxing
    """
    def save_last_image_as_atom_config(self, atom_config_save_path):
        with open(atom_config_save_path, 'w') as wf:
            for j in self.all_image[-1]:
                wf.write(j)
        print("{} saved.".format(atom_config_save_path))

    """
    @Description :
        save images into one MOVEMENT file
        patten: type of write, 'w' or 'a'
    @Returns     :
    @Author       :wuxingxing
    """
    def save_all_image_as_one_movement(self, save_path, interval=1, patten="w"):
        with open(save_path, patten) as wf:
            for i in range(len(self.all_image)):
                if i % interval == 0:
                    for j in self.all_image[i]:
                        wf.write(j)
        print("all images save to {}".format(save_path))

    def save_all_image_as_one_movement_(self, save_path, interval=1, patten="w"):
        with open(save_path, patten) as wf:
            for i in range(len(self.all_image)):
                if i % interval == 0:
                    for j in self.all_image[i]:
                        wf.write(j)
        print("all images save to {}".format(save_path))

    """
    @Description :
        extract images from MOVEMENT file    
    @Returns     :
    @Author       :wuxingxing
    """
    def get_all_image(self):
        all_image = []
        num_image = 0
        atom_nums = []

        file = open(self.MOVEMENT_path,"r")
        lines = file.readlines()
        file.close() 
        mk = -1 
        out_mk = 0 
        raw = []
        singleImage = []  
        for line in lines:
            #encountering a new image 
            if len(line.split())>2 and line.split()[1] == 'atoms,Iteration' and len(singleImage)!=0:
                all_image.append(singleImage.copy())
                singleImage.clear()
            singleImage.append(line)
        all_image.append(singleImage.copy())
        num_image = len(all_image)   
        atom_nums = int(all_image[0][0].split()[0])
        print ("number of total image:",num_image)
        return all_image, num_image, atom_nums

    def get_all_images_etot_force(self, md_type="DFT"):
        images_etot = []
        images_force = []
        for i in range(self.num_image):
            images_etot.append(self.get_image_etot(i, md_type))
            images_force.append(self.get_image_force(i))
        return images_etot, images_force

    '''
    description: get movements energy. For dpkf md, the energy get from the first line 'ep', because the movement file does not have the 'atomic-energy' block.
    param {*} self
    param {*} img_idx 
    param {*} type for dpkf model, it should be 'ML', otherwise 'DFT'
    return {*}
    '''    
    def get_image_etot(self, img_idx, md_type="DFT"):
        if img_idx >= self.num_image:
            raise Exception("input image index too lagre")
        img_atom_num = int(self.all_image[img_idx][0].split()[0]) 
        result = 0.0 
        mk = -1
        if md_type == "ML":
            result = float(self.all_image[img_idx][0].split("=")[2].strip().split()[1])
        else:
            for line in self.all_image[img_idx]:
                # etot section 
                if 'Atomic-Energy,' in line:
                    mk +=1
                    continue 
                if mk >=0 and mk < img_atom_num:
                    result += float(line.split()[1])
                    mk +=1 
        return result

    def get_image_force(self, img_idx): 

        if img_idx >= self.num_image:
            raise Exception("input image index too lagre")
        img_atom_num = int(self.all_image[img_idx][0].split()[0]) 
        mk = -1 
        result = [] 
        g = lambda x:[int(x[0]),float(x[1]),float(x[2]),float(x[3])] 
        for line in self.all_image[img_idx]:
            # etot section 
            if 'Force' in line:
                mk +=1
                continue 
            if mk >=0 and mk < img_atom_num:
                #print (line)
                result.append(g(line.split()))
                mk +=1
        return result
        #print (img_atom_num)

'''
description: 计算两个movement 能量和力误差
return {*}
'''
def cal_loss():
    dpkf_movement_path = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/draw_dir/DFT_DPKF_1000K/dpkf_md/MOVEMENT"
    dpkf = MovementOp(dpkf_movement_path)
    dpkf_energy, dpkf_force = dpkf.get_all_images_etot_force("ML")
    
    dft_movement_path = "/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/draw_dir/DFT_DPKF_1000K/dft_md/MOVEMENT"
    dft = MovementOp(dft_movement_path)
    dft_energy, dft_force = dft.get_all_images_etot_force("DFT")
    
    rmse_energy = []
    rmse_force = []

    from sklearn.metrics import mean_squared_error

    for i in range(len(dft_energy)):
        rmse_energy.append(np.sqrt(mean_squared_error([dpkf_energy[i]], [dft_energy[i]]))/dpkf.atom_nums)
        f_rmse = np.sqrt(mean_squared_error(dpkf_force[i], dft_force[i]))
        f_rmse = f_rmse if f_rmse < 20 else 20
        rmse_force.append(f_rmse)
    
    # print(rmse_energy)
    # print(rmse_force)
    
    from draw_pictures.draw_util import draw_lines
    draw_lines(x_list = [[i for i in range(len(rmse_energy))]], y_list=[rmse_energy], \
        save_path="/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/draw_dir/DFT_DPKF_1000K/rmse_energy.png",
        title="RMSE energy of DFT MD with dpfk MD under (1000K, 2500fs)",
        y_label = "Etot rmse (eV)", # Force rmse (eV/Å)
        x_label = "md steps"
        )
    
    draw_lines(x_list = [[i for i in range(len(rmse_energy))]], y_list=[rmse_force], \
        save_path="/home/wuxingxing/codespace/MLFF_wu_dev/active_learning_dir/cuo_3phases_system/draw_dir/DFT_DPKF_1000K/rmse_force.png",
        title="RMSE force of DFT MD with dpfk MD under (1000K, 2500fs)",
        y_label = "Force rmse (eV/Å)", # 
        x_label = "md steps"
        )

    print()

""" function test"""
def separate_movement():
    movs = ["1500k"]
    for m in movs:
        movement_save_path = "/home/wuxingxing/datas/system_config/cu_4phases_system/atom_configs/slab/PWdata/{}/MOVEMENT".format(m)
        atom_config_save_dir = "/home/wuxingxing/datas/system_config/cu_4phases_system/atom_configs/slab/atom_configs/{}".format(m)
        tmp = MovementOp(movement_save_path)
        tmp.save_each_image_as_atom_config(atom_config_save_dir, ratio=10)

def save_images_as_movement():
    movement_source_path = ["/home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/bulk/MOVEMENT1000k",
                            "/home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/slab/MOVEMENT_100_1500k",
                            "/home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/liquid/MOVEMENT_2000K",
                            "/home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/gas/MOVEMENT1000k"
                            ]
    movement_save_dir = "/home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/100_4phases"
    for movement in movement_source_path:
        tmp = MovementOp(movement)
        save_path = movement_save_dir
        pwdata_path = os.path.join(save_path, "PWdata")
        if os.path.exists(pwdata_path) is False:
            os.makedirs(pwdata_path)
        
        if os.path.exists(os.path.join(pwdata_path, "{}_{}".format(movement.split('/')[-2], movement.split('/')[-1]))) is False:
            os.makedirs(os.path.join(pwdata_path, "{}_{}".format(movement.split('/')[-2], movement.split('/')[-1])))

        tmp.save_images_by_index(os.path.join(pwdata_path, "{}_{}".format(movement.split('/')[-2], movement.split('/')[-1])), 0, 125)

        tmp.save_image_as_atom_config(os.path.join(movement_save_dir, "{}_{}_atom.config".format(movement.split('/')[-2], movement.split('/')[-1])), 0)

    res = os.popen("cp /home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/slab/parameters.py {}".format(save_path))
    res = os.popen("cp /home/wuxingxing/datas/system_config/cu_4phases_system/dpkf_data/slab/gen_data.sh {}".format(save_path))
    cwd = os.getcwd()
    os.chdir(save_path)
    import subprocess
    # result = subprocess.call("bash -i gen_dpkf_data.sh", shell=True)
    result = os.popen("bash -i gen_data.sh")
    # assert(result == 0)
    print(result.readlines())
    print("{} done~".format(save_path))
    os.chdir(cwd)

def sample_movements():
    root_dir = "/share/home/wuxingxing/datas/system_config/cu_72104/dft_test"
    mov_list = ["400k_2p2fs", "800k_2p2fs", "1300k_1ps1fs", "1600k_1ps1fs", "2000k_1ps1fs"]
    # mov_list = ["400k_2p2fs","600k_2p2fs","800k_2p2fs","1000k_2p2fs","1300k_1ps1fs", "1400k_1ps1fs", "1600k_1ps1fs", "1800k_1ps1fs", "2000k_1ps1fs"]

    save_path = os.path.join(root_dir, "400_2000k_5con_movement")

    for mov in mov_list:
        tmp = MovementOp(os.path.join(root_dir, "dft_movement", mov, "MOVEMENT"))
        # tmp.save_all_image_as_one_movement(save_path, interval=10, patten='a')
        tmp.save_images_by_index(save_path, start=700, end=800, patten='a')

def tmp_print_index():
    data_paths = ["/share/home/wuxingxing/datas/system_config/cu_72104/dft_test/5_400_2000k/kpu_data/train"]
    dirs = []
    for data_path in data_paths:
        for current_dir, child_dir, child_file in os.walk(data_path, followlinks=True):
            if len(child_dir) == 0 and "Ri.npy" in child_file:
                dirs.append(current_dir)
    
    print(dirs)
if __name__ == '__main__':
    # parsing args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='MOVEMENT')
    # parser.add_argument('-r', '--ratio', help='specify number of samples per set', type=float, default=0.2)
    # parser.add_argument('-s', '--savepath', help='specify stored directory', type=str, default='.')
    # args = parser.parse_args()

    # # cal_loss()
    # movement_source_path = args.input
    # ratio = args.ratio
    # savepath = args.savepath
    # tmp = MovementOp(movement_source_path)
    # # tmp.separate_movement_2part_by_ratio(savepath, ratio)
    # tmp.cut_movement_by_indexs(savepath, 0, 1000)

    # tmp.save_each_image_as_atom_config(savepath)

    # separate_movement()
    # save_images_by_indexs()
    # save_images_as_movement()
    
    # movement_source_path = "/home/wuxingxing/datas/system_config/cuo_3phases_system/dft_md_bk/MOVEMENT"
    # tmp = MovementOp(movement_source_path)
    # images_etot, images_force = tmp.get_all_images_etot_force()
    # print(images_etot)
    
    sample_movements()
    # tmp_print_index()