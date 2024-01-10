from matersdk.io.publicLayer.structure import DStructure
from matersdk.io.publicLayer.neigh import StructureNeighborsDescriptor
from matersdk.feature.avg.avgbond import AvgBond


### Part I. Custom Parameters
poscar_path = "/data/home/liuhanyu/hyliu/code/matersdk/test_data/xyz/POSCAR.vasp"
atom_config_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
structure = DStructure.from_file(
                file_path=atom_config_path,
                file_format="pwmat")
rcut = 3.2                   # 最大键长 (截断半径)
atomic_number_1 = 42         # 成键原子的原子序数（顺序无所谓）
atomic_number_2 = 16          # 成键原子的原子序数
scaling_matrix = [3, 3, 1]   # 二维材料:[3, 3, 1]; 三维材料:[3, 3, 3]


### Part II. Run the program.
struct_neigh = StructureNeighborsDescriptor.create(
                            'v1',
                            structure,
                            rcut,
                            scaling_matrix,
)
avg_bond_length = AvgBond.get_avg_bond_length(
                struct_neigh=struct_neigh,
                atomic_number_1=atomic_number_1,
                atomic_number_2=atomic_number_2
)
print("在{0}埃内，{1}-{2}的平均键长为{3}埃".format(rcut, atomic_number_1, atomic_number_2, avg_bond_length))
    