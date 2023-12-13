import os 
from active_learning.slurm import SlurmJob
import dpdata

def make_script_dpkit(model_dir, data_dir, out_script_name="dp_model_div_slurm.job", out_file_name="model_devi.out"):
    with open(os.path.join("./template_script_head", "dpkit.job"), 'r') as rf:
        script_head = rf.readlines()
    cmd = ""
    for i in script_head:
        cmd += i
    cmd += "\n"
    cmd += "cd {} \n".format(model_dir)
    
    cmd = "dp model-devi -m graph.000.pb graph.001.pb graph.002.pb graph.003.pb -s {} -o {}\n".format(data_dir, out_file_name)
    with open(out_script_name, "w") as wf:
        wf.write(cmd)
    return out_script_name

def calculate_model_devi_of_traj(model_dir, data_dir, out_script_name="dp_model_div_slurm.job", out_file_name="model_devi.out"):
    out_script_name = make_script_dpkit(model_dir, data_dir, out_script_name="dp_model_div_slurm.job", out_file_name="model_devi.out")
    slurm = SlurmJob()
    slurm.set_cmd("sbatch {}".format(out_script_name))
    slurm.set_tag("1")
    slurm.submit()
    slurm.running_work()

'''
Description: 
test model devi of dpgen, data from movement
Returns: 
Author: WU Xingxing
'''
def test_devi_by_movement():
    model_dir = ""
    data_dir = ""
    out_script_name="dp_model_div_slurm.job"
    out_file_name="model_devi.out"
    calculate_model_devi_of_traj(model_dir, data_dir, out_script_name, out_file_name)

def test_kpu_by_traj():
    # traj to atom.config -> to movement -> features
    # set job
    # run kpu info
    pass

if __name__ == "__main__":
    test_devi_by_movement()


