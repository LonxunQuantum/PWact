import unittest

# python3 -m matersdk.io.pwmat.output.test.test_fatbandstructureTxt
from ..fatabandstructureTxt import FatbandStructure


class FatbandStructureTest(unittest.TestCase):
    def test_fatbandstructure(self):
        fatbandstructure_txt_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf/nonscf/dos/fatbandstructure_1.txt"
        fatbandstructure = FatbandStructure(
                fatbandstructure_txt_path=fatbandstructure_txt_path,
        )        
        
        ### Step 1. 得到能带的条数
        print("\n1. 能带的总数目:", end="\t")
        print(fatbandstructure._get_num_bands())
        
        ### Step 2. 得到 kpoints 的数目
        print("\n2. KPOINTS的总数目:", end="\t")
        print(fatbandstructure._get_num_kpoints())
        
        ### Step 3. 得到 `BAND` 所在行的索引
        print("\n3. BAND 所在行的索引:")
        print(fatbandstructure._get_BAND_mark_idxs())
        
        ### Step 4. 得到体系内的所有元素种类
        print("\n4. 体系内的所有元素种类:")
        print(fatbandstructure._get_elements_lst())

        ### Step 5. preproceee 读取 pd.DataFrame，注意不包括空行的`BAND行`
        print("\n5. 预处理后的 DataFrame:")
        print(fatbandstructure._preprocess())
        
        ### Step 6. 将预处理后的 DataFrame 按照能带分为新的 DataFrames
        print("\n6. 将预处理后的 DataFrame 按照能带分为新的 DataFrames:")
        print(fatbandstructure.get_total_dfs_lst()[0])
        
        ### Step 7. 求元素各轨道的权重之和
        print("\n7. 求元素各轨道的权重之和:")
        print(fatbandstructure.get_element_dfs_lst()[0])
        
        ### Step 8. 求某个轨道的权重
        orbital_name = "mo-4dxy"
        print("\n8. 某个轨道的权重:")
        print(fatbandstructure.get_orbital_dfs_lst(orbital_name=orbital_name)[0])


if __name__ == "__main__":
    unittest.main()