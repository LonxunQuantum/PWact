import os
from typing import List
import copy


class GenVext(object):
    '''
    Description
    -----------
        1. 使用 `add_field.x` 外加电场计算时，同时需要 `gen.vext`
    
    Attributes
    ----------
        1. `self.vr_center`: List[float]
        2. `self.vr_type`: int[1|2|3]
        3. `self.add_vr`: bool
        4. `self.as_param_lst`: List[float]
    '''
    def __init__(
                self,
                vr_center:List[float],
                vr_type:int,
                add_vr:bool,
                *args:List[float],
                ):
        '''
        Description
        -----------
            1. 
        
        `gen.vext` 文件说明
        ------------------
            1. VR_CENTER: 
                - `VR_CENTER = a1 a2 a3`, 外部电场的中心（x, y, z 方向上的分数坐标）
                - 默认值: `0.5 0.5 0.5`
            2. VR_TYPE: 支持三种类型的外部电场，详情参考 `VR_DETAIL`
            3. VR_DETAIL: 
                1. `VR_TYPE = 1`, `VR_DETAIL = a4 a5 a6`
                    - Vext(r) = (x-a1)*a4 + (y-a2)*a5 + (z-a3)*a6
                    - a4, a5, a6 的单位: Hartree/Bohr
                2. `VR_TYPE = 2`, `VR_DETAIL = a4 a5 a6 a7 a8 a9`
                    - Vext(r) = (x-a1)*a4 + (x-a1)^2*a5 + \
                                (y-a2)*a6 + (y-a2)^2*a7 + \
                                (z-a3)*a8 + (z-a3)^2*a9
                    - a4, a6, a8 的单位: Hartree/Bohr
                    - a5, a7, a9 的单位: Hartree/Bohr^2
                3. `VR_TYPE = 3`, `VR_DETAIL = a4 a5`
                    - Vext(r) = a4*exp{-[ (x-a1)^2+(y-a2)^2+(z-a3)^3 ]/a5^2}
                    - a4 的单位: Hartree
                    - a5 的单位: Bohr
            4. ADD_VR: 
                    - `ADD_VR = T`: 将新产生的电场加到输入文件(如`IN.VR`)的势场上面，并输出到EXT文件
                    - `ADD_VR = F`: 仅将新电场输出到EXT文件
        '''
        self.vr_center = vr_center
        self.vr_type = vr_type
        assert (vr_type in [1, 2, 3])
        self.add_vr = add_vr
        args = list(args)
        self.as_param_lst = args
    
    
    def to(self, output_path:str):
        '''
        Description
        -----------
            1. 输出到 `gen.vext` 文件中
        '''
        with open(output_path, "w") as f:
            f.write("VR_CENTER = {0}\n".format(
                        ' '.join([str(value) for value in self.vr_center])
                                )
                    )
            f.write("VR_TYPE = {0}\n".format(self.vr_type))
            f.write("VR_DETAIL = {0}\n".format(
                        ' '.join([str(value) for value in self.as_param_lst])
                                )
                    )
            if self.add_vr:
                add_vr = "T"
            else:
                add_vr = "F"
            f.write("ADD_VR = {0}\n".format(add_vr))