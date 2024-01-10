import torch
import numpy as np
from typing import List

from ....io.publicLayer.structure import DStructure
from ..deepmd.preprocess import InferPreprocessor


class DpInfer(object):
    '''
    Note
    ----
        1. 单卡inference -- "cuda:0"
    '''
    def __init__(
                self,
                pt_file_path:str,
                device:str,
                rcut:float,
                rcut_smooth:float,
                davg:np.ndarray,
                dstd:np.ndarray,
                center_atomic_numbers:List[int],
                nbr_atomic_numbers:List[int],
                max_num_nbrs:List[int],
                scaling_matrix:List[int],
                reformat_mark=True,
                coords_are_cartesian=True,
                ):
        '''
        Description
        -----------
            1. `torch.load(.pt_file)`, then do inference according to DeepPot-SE
        
        Parameters
        ----------
            1. pt_file_path: str
                - The path of `.pt` file of NN model
            1. device: str
                - e.g. "cuda:0"
            1. rcut: float
                - radius cutoff
            2. rcut_smooth: float
                - radius cutoff smooth
            3. davg: np.ndarray
                1) 每个中心原子有一个Rij的平均值(4维np.ndarray)
                2) .shape = (num_elements, 4)
                1) 每个中心原子有一个Rij的方差(4维np.ndarray)
                2) .shape = (num_elements, 4)
            5. center_atomic_numbers: List[int]
                1) 中心原子的原子序数
            6. nbr_atomic_numbers: List[int]
                1) 近邻原子的原子序数
            7. max_num_nbrs: List[int]
                1)  近邻原子的最大近邻数，需要与 `nbr_atomic_numbers` 对应
            8. scaling_matrix: List[int]
                1) 取值
                    - 三维材料: [3, 3, 3]
                    - 二维材料: [3, 3, 1]         
            9. reformat_mark: bool, default=True
            10. coords_are_cartesian: bool, default=True
        '''
        self.pt_file_path = pt_file_path
        self.device = device
        self.rcut = rcut
        self.rcut_smooth = rcut_smooth
        self.davg = davg
        self.dstd = dstd
        self.center_atomic_numbers = center_atomic_numbers
        self.nbr_atomic_numbers = nbr_atomic_numbers
        self.max_num_nbrs = max_num_nbrs
        self.scaling_matrix = scaling_matrix
        self.reformat_mark = reformat_mark
        self.coords_are_cartesian = coords_are_cartesian
        
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available, please change device.")
    
    
    def _preprocess(self, structure:DStructure):
        infer_preprocessor = InferPreprocessor(
                structure=structure,
                rcut=self.rcut,
                rcut_smooth=self.rcut_smooth,
                scaling_matrix=self.scaling_matrix,
                davg=self.davg,
                dstd=self.dstd,
                center_atomic_numbers=self.center_atomic_numbers,
                nbr_atomic_numbers=self.nbr_atomic_numbers,
                max_num_nbrs=self.max_num_nbrs,
                reformat_mark=self.reformat_mark,
                coords_are_cartesian=self.coords_are_cartesian
        )
        
        ### Step 1. 获取 model.forward() 的参数, 并且 to_tensor
        ### Step 1.1. 获取 ImageDR
        rc:torch.Tensor = torch.from_numpy(
                infer_preprocessor.expand_rc()
        ).double().to(self.device).requires_grad_()
        ### Step 1.2. 获取 Rij, Rij_d
        tildeR, tildeR_deriv = infer_preprocessor.expand_tildeR()
        tildeR:torch.Tensor = torch.from_numpy(tildeR).double().to(self.device).requires_grad_()
        tildeR_deriv:torch.Tensor = torch.from_numpy(tildeR_deriv).double().to(self.device).requires_grad_()
        ### Step 1.3. 获取 list_neigh
        list_neigh:torch.Tensor = torch.from_numpy(
                    infer_preprocessor.expand_list_neigh()
        ).double().to(self.device).requires_grad_()
        ### Step 1.4. 获取 natoms
        natoms_image:np.ndarray = infer_preprocessor.expand_natoms_img()

        return rc, tildeR, tildeR_deriv, list_neigh, natoms_image
    

    def infer(self, structure:DStructure):
        model = torch.load(f=self.pt_file_path)
        print("model.to : ", self.device)
        model.to(self.device)
        model.eval()
        
        ### Step 1. Get input for `model.forward()`
        rc, tildeR, tildeR_deriv, list_neigh, natoms_image = \
                        self._preprocess(structure=structure)
        
        ### Step 2. Do calculation
        print(model)
        e_tot, e_atoms, f_atoms, virial = model(rc, tildeR, tildeR_deriv, list_neigh, natoms_image)
         
        ### Step 3. cuda -> cpu; tensor -> ndarray
        if "cuda" in self.device:
            e_tot = np.squeeze( e_tot.cpu().detach().numpy(), axis=0 )
            e_atoms = np.squeeze( e_atoms.cpu().detach().numpy(), axis=0 )
            f_atoms = np.squeeze( f_atoms.cpu().detach().numpy(), axis=0 )
            virial = virial.cpu().detach().numpy().reshape(3, 3)
        elif "cpu" in self.device:
            e_tot = np.squeeze( e_tot.detach().numpy(), axis=0 )
            e_atoms = np.squeeze( e_atoms.detach().numpy(), axis=0 )
            f_atoms = np.squeeze( f_atoms.detach().numpy(), axis=0 )
            virial = virial.cpu().detach().numpy().reshape(3, 3)
        
        return e_tot, e_atoms, f_atoms, virial