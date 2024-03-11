import time
import random

def make_iter_name(iter_index: int) :
    iter_format = "%04d"
    return "iter." + (iter_format % iter_index)

def get_iter_from_iter_name(iter_name: str):
    return int(iter_name.split(".")[1].strip())

def make_train_name(model_index: int):
    train_name = "%03d"
    return "train."+(train_name % model_index)

def make_md_name(md_index:int):
    md_name = "%03d"
    return "md."+(md_name % md_index)

def make_md_sys_name(md_index:int, sys_index:int, len_char:int = None):
    md_name = "%03d"
    sys_name = "%03d" if len_char is None else "%0{}d".format(len_char)
    return "md."+(md_name % md_index)+".sys."+(sys_name % sys_index)

def make_temp_press_name(md_index:int, sys_index:int, temp_index:int, press_index:int, len_char:int = None):
    md_name = "%03d"
    sys_name = "%03d" if len_char is None else "%0{}d".format(len_char)
    p_name = "%03d"
    t_name = "%03d"
    return "md."+(md_name % md_index)+".sys."+(sys_name % sys_index)+\
        ".t."+ (sys_name % temp_index) + ".p."+ (p_name % press_index)

def make_temp_name(md_index:int, sys_index:int, temp_index:int, len_char:int = None):
    md_name = "%03d"
    sys_name = "%03d" if len_char is None else "%0{}d".format(len_char)
    t_name = "%03d"
    return "md."+(md_name % md_index)+".sys."+(sys_name % sys_index)+\
        ".t."+ (sys_name % temp_index)

def get_sub_md_sys_template_name():
    return "md.*.sys.*/md.*.sys.*"

def get_md_sys_template_name():
    return "md.*.sys.*"

def get_traj_file_name(traj_index:int):
    return "{}_traj.lammps".format(traj_index)

def make_scf_name(scf_index:int):
    scf_name = "%04d"
    return "scf."+(scf_name % scf_index)

def get_seed_by_time():
    # seed = int(time.time())
    # random.seed(seed)
    return random.randint(1,10000)
    