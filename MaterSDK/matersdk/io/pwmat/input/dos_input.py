from typing import List


class DosInput(object):
    '''
    Description
    -----------
        1. 使用 `plot_DOS.interp.x` 的时候，需要有 `DOS.input` 作为输入
        2. `DOS.input` 的内容:
            ```
            0   # setting 0 -> for all atoms; setting 1 -> for partial atoms. 
            0   # setting 1 -> using interpolation; setting 0 -> using Gaussian broadening
            0.05    4000    #  energy smearing, in eV;  number of energy grid points, default is 4000
            8   8   8   # interpolation grid
            ```
    '''
    def __init__(
            self,
            mark_atoms:int=0,
            mark_method:int=0,
            ismear:float=0.04, num_energies:int=4000,
            grid_interp:List[float]=[6, 6, 6],
            ):
        self.mark_atoms = mark_atoms
        self.mark_method = mark_method
        self.ismear = ismear
        self.num_energies = num_energies
        self.grid_interp = grid_interp
        
    
    def to(self, output_path:str):
        '''
        Desscription
        ------------
            1. 生成 `DOS.input`
        '''
        with open(output_path, "w") as f:
            f.write("{0}  # 设置为0，则绘制所有原子的DOS；设置为1，则绘制部分原子的DOS，需要在atom.config文件POSITION部分的第8列设置原子权重\n".format(self.mark_atoms))
            f.write("{0}  # 设置为1，绘制DOS时使用插值；设置为0，使用高斯展宽(Gaussian broadening)，不做插值\n".format(self.mark_method))
            f.write("{0:<.5f}  {1:<10d} # 做插值的imsearing energy (单位：eV); 能量网格点数，默认为4000\n".format(self.ismear, self.num_energies))
            f.write("{0}  # NM1,NM2,NM3 插值的格子密度，每个格子包含在NQ1, NQ2, NQ3中\n".format(
                        " ".join([str(value) for value in self.grid_interp])
                        )
            )
            
            f.write("0  # 只对TDDFT DOS绘制有用")
            