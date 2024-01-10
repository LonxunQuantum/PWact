from typing import List


class Locator(object):
    @staticmethod
    def locate_all_lines(file_path:str, content:str):
        pass


class LineLocator(Locator):
    @staticmethod
    def locate_all_lines(file_path:str, content:str):
        '''
        Description
        -----------
            1. 定位某段文本所在的行 (返回所有行数)

        Parameters
        ----------
            1. file_path: str
                文件的绝对路径
            2. content: str
                需要定位的内容
        
        Note
        ----
            1. content 必须为大写
            2. 返回的行从 1 开始，便于与 `linecache.getline()` 的对接
        '''
        row_idxs_lst = []
        row_no = 0

        with open(file_path, "r") as f:
            for row_content in f:
                row_no += 1
                
                if content in row_content.upper():
                    row_idxs_lst.append(row_no)
        
        return row_idxs_lst
    

class ListLocator(Locator):
    @staticmethod
    def locate_all_lines(strs_lst:List[str], content:str):
        '''
        Description
        -----------
            1. 定位
            
        Parameters
        ----------
            1. strs_lst: str
                - str 组成的列表
            2. content: str
                - 需要定位的内容
        
        Note
        ----
            1. content 必须大写
            2. 返回的索引从 0 开始，便于与 `列表的索引` 对齐
        '''
        str_idxs_lst = []
        str_no = -1
        
        for tmp_str in strs_lst:
            str_no += 1
            
            if content in tmp_str.upper():
                str_idxs_lst.append(str_no)
        
        return str_idxs_lst