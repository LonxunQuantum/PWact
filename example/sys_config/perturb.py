from matersdk.adalearn.generator.perturbation import BatchPerturbStructure
import os
if __name__ == "__main__":

    """
    1. only need to run once!!!
    2. perturb the structure
    3. seed for adaptive sampling
    """

    Perturbed = ['tmp']
    pert_num = 50
    cell_pert_fraction = 0.03
    atom_pert_distance = 0.01

    BatchPerturbStructure.batch_perturb(
        Perturbed=Perturbed,
        pert_num=pert_num,
        cell_pert_fraction=cell_pert_fraction,
        atom_pert_distance=atom_pert_distance,
    )


    aimd_directory = os.path.join(os.path.abspath(Perturbed[0]), 'AIMD')
    if not os.path.exists(aimd_directory):
        os.makedirs(aimd_directory)

    # Create 'md-0', 'md-1', ..., 'md49' directories under 'AIMD' directory
    for i in range(pert_num):
        md_directory = os.path.join(aimd_directory, f'md-{i}')
        if not os.path.exists(md_directory):
            os.makedirs(md_directory)

        # Link the corresponding config file from 'structures' directory to 'md-{i}' directory
        config_file = os.path.join(os.path.abspath(Perturbed[0]), 'structures', f'{i}.config')
        link_file = os.path.join(md_directory, 'atom.config')
        if os.path.islink(link_file):
            os.remove(link_file)
        os.symlink(config_file, link_file)
