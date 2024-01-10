import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ...io.publicLayer.structure import DStructure



class KPointsSampler(object):
    '''
    Description
    -----------
        1. k-point mesh of the Brillouin zone generated taken into account symmetry.
        2. The method returns the `irreducible kpoints` of the mesh and their weights.

    Attributions
    ------------
        1. structure: DStructure
            需要被分析的结构对象
        2. kmesh: np.array
            KMesh 的取值， e.g. np.array([8, 8, 8])
        3. is_shift: np.array
            是否需要移动
        4. symprec: float
            Tolerance for symmetry finding.
        5. angle_tolerance:
            Angle tolerance for symmetry finding.
    
    Note
    ----
        1. 根据空间群对称性消除重复的 KPoints
    '''
    def __init__(self,
                structure:DStructure,
                kmesh:np.array,
                is_shift:np.array,
                symprec:float=1e-3,
                angle_tolerance:float=5.0,
                ):
        '''
        Parameters
        ----------
            1. structure: DStructure
                需要被分析的结构对象
            2. kmesh: np.array
                KMesh 的取值， e.g. np.array([8, 8, 8])
            3. is_shift: np.array
                是否需要移动
            4. symprec: float
                Tolerance for symmetry finding.
            5. angle_tolerance:
                Angle tolerance for symmetry finding.
        '''
        self.kmesh = kmesh
        self.is_shift = is_shift

        # SpacegroupAnalyzer 的初始化
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.spacegroup_analyser = SpacegroupAnalyzer(
                                        structure=structure,
                                        symprec=symprec,
                                        angle_tolerance=angle_tolerance,
                                        )
    

    def get_kpoints(self):
        '''
        Description
        -----------
            1. 根据 kmesh 选出 kpoints，然后`根据对称性`消除重复的 kpoints
        '''
        return self.spacegroup_analyser.get_ir_reciprocal_mesh(
                                            mesh=self.kmesh,
                                            is_shift=self.is_shift,
                                            )


    def get_num_kpoints(self):
        '''
        Description
        -----------
            1. 返回 KPoints 的数目
        '''
        num_kpoints = len(self.spacegroup_analyser.get_ir_reciprocal_mesh(
                                            mesh=self.kmesh,
                                            is_shift=self.is_shift,
                                            )
                        )
        return num_kpoints