import glob
import os
import math

def split_train_dir(root_dir, split_size=500):
    data_path = os.path.join(root_dir, "train")
    data_res = []
    images = glob.glob(os.path.join(data_path, "image*"))
    images = [_.split('/')[-1] for _ in images]
    images = sorted(images, key=lambda x: int(x.split('_')[1]))
    images_list = split_images(images, split_size)
    
    davg_path = os.path.join(data_path, "davg.npy")
    dstd_path = os.path.join(data_path, "dstd.npy")

    for i, ilist in enumerate(images_list):
        new_data_dir = os.path.join(root_dir, "train_{}".format(i))
        new_dir = os.path.join(new_data_dir, "train")
        if os.path.exists(new_dir) is False:
            os.makedirs(new_dir)
        #link images
        for image in ilist:
            tar_image_dir = os.path.join(new_dir, image)
            if os.path.exists(tar_image_dir) is False:
                os.symlink(os.path.join(data_path, image), tar_image_dir)
        
        #link davg and dstd
        tar_davg_path = os.path.join(new_dir, 'davg.npy')
        if os.path.exists(tar_davg_path) is False:
            os.symlink(davg_path, tar_davg_path)
        tar_dstd_path = os.path.join(new_dir, 'dstd.npy')
        if os.path.exists(tar_dstd_path) is False:
            os.symlink(dstd_path, tar_dstd_path)
        
        #link valid dir
        tar_valid_path = os.path.join(new_data_dir, 'valid')
        if os.path.exists(tar_valid_path) is False:
            os.symlink(os.path.join(root_dir, "valid"), tar_valid_path)
        data_res.append(new_data_dir)
    return data_res

def split_images(images, split_size):
    start = 0
    res = []
    while start < len(images):
        end = start+split_size if start+split_size < len(images) else len(images)
        res.append(images[start:end])
        start += split_size
    return res

'''
Description: 
res_slurm_job get the script of kpu slurm jobs;
the res_done containt the kpu slurm jobs done before.
param {*} dir
Returns: 
Author: WU Xingxing
'''
def get_kpu_slurm_scripts(dir):
    #md_kpu_success_5.tag md_kpu_slurm_5.job
    slurm_job_files = glob.glob(os.path.join(dir, "md_kpu_slurm_*.job"))
    slrum_indexs = [int(job.split('/')[-1].split('.')[0].split('_')[3]) for job in slurm_job_files]

    slurm_job_done_tag = glob.glob(os.path.join(dir, "model_dir", "md_kpu_success_*.tag"))
    tag_indexs = [int(job.split('/')[-1].split('.')[0].split('_')[3]) for job in slurm_job_done_tag]

    res_slurm_job = []
    res_tag = []
    res_done = []
    for i, v in enumerate(slrum_indexs):
        if v in tag_indexs:
            res_done.append(slurm_job_files[i])
        else:
            res_slurm_job.append(slurm_job_files[i])
            res_tag.append(os.path.join(dir, "model_dir", "md_kpu_success_{}.tag".format(v)))
            
    return res_slurm_job, res_tag, res_done

if __name__ == "__main__":
    res = split_train_dir("/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0001/exploring/md_dpkf_dir", 100)
    print(res)