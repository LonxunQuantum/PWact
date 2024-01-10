import os 
import pandas as pd


class Dostotalspin(object):
    def __init__(
                self,
                dos_totalspin_path:str):
        self.dos_totalspin_path = dos_totalspin_path
        self.bak_dosfile_path = self._preprocess()  # 删除了原DOS.totalspin第一行的"#"
    
    
    def __del__(self):
        '''
        Description
        -----------
            1. `Dostotalspin` 对象销毁时，自动删除 `self.bak_dosfile_path` 文件
        '''
        os.remove(self.bak_dosfile_path)
    
        
    def _preprocess(self):
        '''
        Description
        -----------
            Step 1. 删除 DOS.totalspin, DOS.spinup, DOS.spindown 行首的 "#"
            Step 2. 写入新的 DOS.totalspin 到 `DOS.totalspin.bak`
        '''
        ### Step 1. 删除 DOS.totalspin, DOS.spinup, DOS.spindown 行首的 "#"
        with open(self.dos_totalspin_path, "r") as f:
            lines_lst = f.readlines()
            first_line = lines_lst[0]
            first_line_lst = first_line.split()
            try:
                first_line_lst.remove("#")
            except:
                pass
            new_first_line = "\t   ".join(first_line_lst)
            new_first_line += "\n"   # 'Energy\tTotal\tMo-s\tMo-p\tMo-s\tMo-d\tS-s\tS-p\n'

            ### Step 1.1. 删除原来的行首
            lines_lst.pop(0)
            ### Step 1.2. 添加新的行首
            lines_lst.insert(0, new_first_line)
        
        
        ### Step 2. 写入新的 DOS.totalspin 到 `DOS.totalspin.bak`
        folder_path = os.path.dirname(self.dos_totalspin_path)
        filename = os.path.basename(self.dos_totalspin_path)
        bak_dosfile_path = os.path.join(folder_path, "{0}.bak".format(filename))
        with open(bak_dosfile_path, "w") as f:
            f.writelines(lines_lst)
        
        return bak_dosfile_path
    

    def get_tdos(self):
        '''
        Description
        ----------- 
            1. 读取 DOS.totalspin, DOS.spinup, DOS.spindown
        
        Return
        ------
            1. df_tdos: pd.DataFrame
            
        '''
        df_dos = pd.read_csv(
                    self.bak_dosfile_path,
                    delimiter='\s+',
                    )
        try: 
            df_tdos = df_dos.loc[:, ["Energy", "Total"]]
        except KeyError:
            df_tdos = df_dos.loc[:, ["Energy", "total"]]
        
        return df_tdos

    
    def get_pdos_elements(self):
        '''
        Description
        -----------
            1. 投影态密度到各个元素
            
            
        Return
        ------
            1. df_pdos_elements: pd.DataFrame
                       Energy    S   Mo
                0    -65.1360  0.0  0.0
                1    -65.1190  0.0  0.0
                2    -65.1010  0.0  0.0
                3    -65.0830  0.0  0.0
                4    -65.0650  0.0  0.0
                ...
                ...
        '''
        ### Step 1. 读取 DOS.totalspin 文件
        df_dos = pd.read_csv(
                    self.bak_dosfile_path,
                    delimiter='\s+',
                    )
        
        ### Step 2. 查看体系中的所有元素 -- elements_lst
        columns_df = df_dos.columns.to_list()[2:]
        raw_elements_lst = [element_orbital.split('-')[0] for element_orbital in columns_df]
        elements_lst = list(set(raw_elements_lst))
        
        ### Step 3. 将 elements_lst 中元素的各个轨道的 density 相加
        df_elements_lst = []
        new_columns_lst = []
        for tmp_element in elements_lst:
            tmp_mask = df_dos.columns.str.startswith(tmp_element)
            df_tmp_element = df_dos.loc[:, tmp_mask].sum(axis=1)
            df_elements_lst.append(df_tmp_element)
            new_columns_lst.append(tmp_element)
        df_energy = df_dos.loc[:, "Energy"]
        ### Step 3.2. 将 Energy 的 pd.Series 插入到元素`态密度pd.Series`(`df_elements`)前
        df_elements_lst.insert(0, df_energy)
        new_columns_lst.insert(0, "Energy")
        ### Step 3.3. 合并所有元素、能量的pd.Series
        df_pdos_elements = pd.concat(df_elements_lst, axis=1)
        ### Step 3.3. 使用新的columns -- `new_columns_lst`
        df_pdos_elements.columns = new_columns_lst
        
        return df_pdos_elements

    
    def get_pdos_orbitals(self):
        '''
        Description
        -----------
            1. 投影态密度到各个轨道
        
        Note
        ----
            1. 需要使用 `plot_DOS_interp.x` 得到 `DOS.totalspin_projected`
                - `plot_DOS_interp.x` 需要 输入文件 `DOS.input`
0/1    # All atoms or Partial atoms
0/1    # Whether carry out interpolation (0: no;  1: yes)
0.05    4000    # 0.05:     ; 4000: 能量点
        '''
        ### Step 1. 读取 DOS.totalspin 文件
        df_dos = pd.read_csv(
                    self.bak_dosfile_path,
                    delimiter='\s+',
                    )
        try:
            df_pdos_orbitals = df_dos.drop(labels=["Total"], axis=1)
        except KeyError:
            df_pdos_orbitals = df_dos.drop(labels=["total"], axis=1)
        
        return df_pdos_orbitals