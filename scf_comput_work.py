import os
import shutil
def costruct_work(soudir, destdir):
    UPF = "/share/home/wuxingxing/datas/system_config/NCPP-SG15-PBE/Cu.SG15.PBE.UPF"
    etot_input = "/share/home/wuxingxing/codespace/active_learning_mlff/etot.input"
    index = list(range(0,249,5))
    if os.path.exists(destdir) is False:
        os.makedirs(destdir)
    for i in index:
        sou_atom_config = os.path.join(soudir, "atom_{}.config".format(i))
        dest_dir = os.path.join(destdir, "{}".format(i))
        if os.path.exists(dest_dir) is False:
            os.mkdir(dest_dir)
        shutil.copy(sou_atom_config, os.path.join(dest_dir, "atom.config"))
        shutil.copy(UPF, dest_dir)
        shutil.copy(etot_input, dest_dir)



if __name__ == "__main__":
    atom_source = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/exploring/md_traj_dir"
    dest = "/share/home/wuxingxing/datas/al_dir/cu_bulk_system/iter.0000/400k_scf"
    costruct_work(atom_source, dest)