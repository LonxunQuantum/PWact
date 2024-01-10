import unittest
import numpy as np

# python3 -m matersdk.io.publicLayer.test.test_Structure
from ..structure import DStructure
from ..neigh import StructureNeighborsDescriptor
from ...pwmat.output.movement import Movement



class StructureTest(unittest.TestCase):
    def structure_init(self):
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        file_format = "pwmat"
        coords_are_cartesian = False
        
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        #print(structure.cart_coords)
        #print(np.max(structure.cart_coords[:, -1]))
        #print(np.min(structure.cart_coords[:, -1]))
        #print(structure.atoms_lst)
        #print(structure.atomic_numbers_lst)
    
    
    def from_str(self):
        #movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo1/PWdata/data1/MOVEMENT"
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        idx_frame = 2 # 帧数从 0 开始计数
        
        ### Step 0. 得到某 frame 的 string
        atom_config_string = movement._get_frame_str(idx_frame=idx_frame)
       
        ### Step 1. 
        structure = DStructure.from_str(
                            str_content=atom_config_string,
                            str_format="pwmat",
                            coords_are_cartesian=False)
        print(structure)
    

    def structure_to_(self):
        file_format = "pwmat"
        # 1. 普通的测试
        file_path = "/Users/mac/我的文件/Mycode/new/new2/matersdk/test_data/atom_config/atom.config"
        # 2. 测试 DStructure.site_properties["magnetic_properties"]
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        
        # 1. 
        output_file_path = \
                "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/test.config"
        # 2. 
        output_file_path = \
                "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/POSCAR"
        coords_are_cartesian = False

        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        structure.reformat_elements_()

        #print(structure.site_properties["magmom"])

        # 1.
        #structure.to(output_file_path=output_file_path,
        #            output_file_format="rndstr.in")
        
        # 2. 
        structure.to(output_file_path=output_file_path,
                    output_file_format="vasp",
                    include_magnetic_moments=True,
                   )


    def judge_vacuum_exist(self):
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False

        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )

        #print(structure.judge_vacuum_exist())
    
    
    def reformat_elements(self):
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        elements_lst = ["S", "Mo"]
        
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        
        #print("Step 1. Primimary structure:")
        #print(structure)
        new_structure = structure.reformat_elements(elements_lst=elements_lst)
        #print("\nStep 2. New structure:")
        #print(new_structure)  
    
    
    def remove_vacanies(self):
        file_format = "pwmat"
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        
        #print("删除体系内所有空位前，体系内的原子数：{0}".format(
        #                                len(structure.atomic_numbers)
        #                                )
        #      )
        #structure.remove_vacanies()
        #print("删除体系内所有空位前，体系内的原子数：{0}".format(
        #                                len(structure.atomic_numbers)
        #                                )
        #      )
        #print("第 3 个 site 处的magmom: {0}".format(structure.sites[2].magmom)
        #      )
        #print(structure)
        #structure.make_supercell([2,2,2])
        #print(structure)
    
    
    def make_supercell_(self):
        file_format = "pwmat"
        # 1. 普通的测试
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        scaling_matrix = np.array([3,3,3])
        
        supercell = structure.make_supercell_(
                                scaling_matrix=scaling_matrix,
                                )
        #print(supercell)
        #print(supercell)
        #supercell.to(output_file_path="./POSCAR",
        #            output_file_format="vasp",
        #            include_magnetic_moments=True,
        #           )
        
        
    def get_bidx2aidx_supercell(self):
        file_format = "pwmat"
        # 1. 普通的测试
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        scaling_matrix=np.array([3, 3, 1])
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        bidx2aidx = structure._get_bidx2aidx_supercell(scaling_matrix=scaling_matrix)
        #print(bidx2aidx)

    def get_key_idxs(self):
        file_format = "pwmat"
        # 1. 普通的测试
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        scaling_matrix=np.array([3, 3, 1])
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        
        key_idxs = structure.get_key_idxs(scaling_matrix=scaling_matrix)
        #print(key_idxs)
        
 
    def get_site_index(self):
        file_format = "pwmat"
        # 1. 初始化结构
        file_path = "/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config"
        coords_are_cartesian = False
        structure = DStructure.from_file(
                        file_path=file_path,
                        file_format=file_format,
                        coords_are_cartesian=coords_are_cartesian
                        )
        
        # 2. 找到近邻原子
        scaling_matrix=np.array([3, 3, 1])
        rcut = 3.2
        reformat_mark = True
        coords_are_cartesian = True   
        neighbors_v1 = StructureNeighborsDescriptor.create(
                        "v1",
                        structure=structure,
                        rcut=rcut,
                        scaling_matrix=scaling_matrix,
                        reformat_mark=reformat_mark,
                        coords_are_cartesian=coords_are_cartesian,
                        )
        
        # 3. Run test function
        site_coord = structure.sites[-2].coords + structure.lattice.matrix[0]
        
        #print("Step 1. The corresponding index in primitive cell is :",)
        #print( structure.get_site_index(site_coord=site_coord) )
        
        print(structure)
        print("Step 2.")
        for tmp_idx, tmp_coord in enumerate(neighbors_v1.key_nbr_coords[0]):
            # 一个S周围有3个Mo、6个S
            print(neighbors_v1.key_nbr_distances[0][tmp_idx], structure.get_site_index(site_coord=tmp_coord))


        #print("Step 3.")
        #tmp_coord = neighbors_v1.key_nbr_coords[0][3]
        #print(tmp_coord)
        ##print(tmp_coord)
        #structure.get_site_index(site_coord=tmp_coord)


    def get_centroid(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure = movement.get_frame_structure(idx_frame=0)
        print("The cooordinate of structure is : ")
        print(structure.get_centroid())
    
    
    def test_get_natoms(self):
        movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
        movement = Movement(movement_path=movement_path)
        structure = movement.get_frame_structure(idx_frame=0)
        
        print("Step 1. 原子数目按照默认顺序 (从小到大排列):", end='\t')
        print(structure.get_natoms())
        
        print("Step 2. 原子数目按照 [14, 3] 顺序排列:", end='\t')
        print(structure.get_natoms(atomic_numbers_order=[14, 3]))


if __name__ == "__main__":
    unittest.main()