import re
import pandas as pd
import numpy as np
from io import StringIO
from ..utils.lineLocator import LineLocator


class FatbandStructure(object):
    '''
    Description
    -----------
        1. 使用 `plot_fatbandstructure.x` 之后，会产生 `fatbandstruture_1.txt` 文件，
           此 class 用于解析这个文件的各类信息
    
    Parameters
    ----------
        1. fatbandstructure_txt_path: str
            - `fatbandstructure_1.txt` 文件的路径
        2. KPOINT 的距离单位转换为`埃`
    '''
    BOHR = 0.529177249
    
    def __init__(
                self,
                fatbandstructure_txt_path:str
                ):
        self.fatbandstructure_txt_path = fatbandstructure_txt_path
        self.num_bands = self._get_num_bands()
        self.num_kpoints = self._get_num_kpoints()
        self.elements_lst = self._get_elements_lst()

    
    def _get_num_bands(self):
        '''
        Description
        -----------
            1. 得到能带的数目
        '''
        idx_lines_lst = LineLocator.locate_all_lines(
                    file_path=self.fatbandstructure_txt_path,
                    content="BAND"
                    )
        return len(idx_lines_lst)

    
    def _get_num_kpoints(self):
        '''
        Description
        -----------
            1. 得到 kpoints 的数目
        '''
        idx_lines_lst = LineLocator.locate_all_lines(
                    file_path=self.fatbandstructure_txt_path,
                    content="BAND",
        )
        num_kpoints = idx_lines_lst[1] - idx_lines_lst[0] - 2
        return num_kpoints
    

    def _get_BAND_mark_idxs(self):
        '''
        Description
        -----------
            1. 找到 `BAND` 所在的行，并返回所在行的索引
        
        Note
        ----
            1. 索引是从 1 开始的，便于 `linecache.getline()` 调用
        '''
        idx_lines_lst = LineLocator.locate_all_lines(
                file_path=self.fatbandstructure_txt_path,
                content="BAND"
        )
        return idx_lines_lst


    def _get_elements_lst(self):
        '''
        Description
        -----------
            1. 找到本体系中所哟元素并返回
        '''
        df = self._preprocess()
        cols_lst_ = list(df.columns)
        cols_lst_.remove('KPOINT')
        cols_lst_.remove('ENERGY')
        cols_lst_.remove('weight_tot')
        
        cols_lst = list( set([entry.split('-')[0] for entry in cols_lst_])
        )
        
        return cols_lst
    

    def _preprocess(self):
        '''
        Description
        -----------
            1. 预处理，将 `fatbandstructure_1.txt` 读取成 pd.DataFrame，
            2. 例如有64条能带、29个kpoint的能带结构，会产生 64*29=1856 columns
        
        Note
        ----
            1. 跳过 `BAND 行` 和 `空行`
        '''
        ### Step 1. 跳过 `BAND 行` 和 `空行`
        ss = StringIO()
        with open(self.fatbandstructure_txt_path, 'r') as f:
            for line in f:
                if (line=='' or "BAND" in line):
                    continue
                else:
                    ss.write(line)
        ss.seek(0)   # "rewind" to the beginning of the StringIO object
        
        df = pd.read_csv(
                        ss,
                        delimiter='\s+'
        )
        df.loc[:, "KPOINT"] = df.loc[:, "KPOINT"] / self.BOHR
       
        return df
    
    
    def get_total_dfs_lst(self):
        '''
        Description
        -----------
            1. 将预处理后(经历`self._preprocess()`)后的 pd.DataFrame 按照
               不同能带分为新的 DataFrames，并组成列表
            
        Return
        ------
            1. dfs_lst: List[pd.DataFrame]
                - 每个 DataFrame 代表一条能带
        '''
        df = self._preprocess()
        ### Step 1. 每条能带组成了新的 DataFrame
        dfs_lst = np.array_split(df, self.num_bands)
        
        return dfs_lst
    
    
    def get_element_dfs_lst(self):
        '''
        Description
        -----------
            1. 
            
        Return
        ------
                    KPOINT   ENERGY  weight_tot        Mo         S
            0     0.000000 -62.9050         1.0  0.996537  0.003462
            1     0.063172 -62.9040         1.0  0.996569  0.003436
            2     0.126341 -62.9030         1.0  0.996617  0.003383
        '''
        df = self._preprocess()
        ### Step 1. 取出 `KPOINTS`, `ENERGY`, `weight_tot` 对应的列 (type=pd.Series)
        df_head = df.loc[:, ["KPOINT", "ENERGY", "weight_tot"]]
        series_element_lst = []
        
        ### Step 2. 按照元素求权重的和
        for tmp_element in self.elements_lst:
            tmp_df_element = df.filter(regex="^{0}|KPOINT|ENERGY|weight_tot".format(tmp_element))
            tmp_serie_element = tmp_df_element.filter(regex="^{0}".format(tmp_element)).sum(axis=1)
            series_element_lst.append( tmp_serie_element )
        ### 将 `KPOINTS`, `ENERGY`, `weight_tot` 与各元素权重和拼成一个 pd.DataFrame
        series_element_lst.insert(0, df_head)
        new_cols_lst = ["KPOINT", "ENERGY", "weight_tot"] + self.elements_lst
        new_df = pd.concat(series_element_lst, axis=1)
        ### 赋予新的 columns
        new_df.columns =  new_cols_lst
        
        ### Step 3. 将不同能带的能带分成不同的 pd.DataFrame
        element_dfs_lst = np.array_split(new_df, self.num_bands)
        return element_dfs_lst
        
        
    def get_orbital_dfs_lst(self, orbital_name:str):
        '''
        Description
        -----------
            1. 获取某个 orbital 的权重
        
        Parameters
        ----------
            1. orbital_name: str
                - e.g. Mo-4Dxy (大小写无所谓)
        '''
        df = self._preprocess()
        re_pattern = re.compile("{0}|KPOINT|ENERGY|weight_tot".format(orbital_name), re.IGNORECASE)
        df_return = df.filter(regex=re_pattern)
        
        ### Step 2. 将不同band，分成不同的 pd.DataFrame
        orbital_dfs_lst = np.array_split(df_return, self.num_bands)
        
        return orbital_dfs_lst