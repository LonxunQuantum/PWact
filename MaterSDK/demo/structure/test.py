from matersdk.io.publicLayer.structure import DStructure

structure = DStructure.from_file(
                file_format="pwmat", 
                file_path="/data/home/liuhanyu/hyliu/code/matersdk/demo/structure/atom.config")
supercell = structure.make_supercell_(scaling_matrix=[3, 3, 1])
print(supercell.cart_coords)