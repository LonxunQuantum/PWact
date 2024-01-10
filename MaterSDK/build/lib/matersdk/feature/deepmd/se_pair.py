import numpy as np
from abc import ABC, abstractclassmethod

from ...io.publicLayer.neigh import StructureNeighborsV1
from .premise import DpFeaturePairPremiseDescriptor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in reciprocal")


class DpseTildeRPairBase(ABC):
    @abstractclassmethod
    def _get_premise(self):
        pass
    
    @abstractclassmethod
    def _get_s(self):
        pass
    
    @abstractclassmethod
    def _get_tildeR(self):
        pass
    


class DpseTildeRPairDescriptor(object):
    registry = {}
    
    @classmethod
    def register(cls, name:str):
        def wrapper(subclass:DpseTildeRPairBase):
            cls.registry[name] = subclass
        return wrapper

    @classmethod
    def create(cls, name:str, *args, **kwargs):
        subclass = cls.registry[name]
        if subclass is None:
            raise ValueError(f"No DpseTildeRPair registered with name '{name}'")
        return subclass(*args, **kwargs)
    
    
    
@DpseTildeRPairDescriptor.register("v1")
class DpseTildeRPairV1(DpseTildeRPairBase):
    '''
    Description
    -----------
        1. In `DeepPot-SE`, features is constructed as: 
            $D^{i} = (g^{i1})^T \widetilde{R}^i (\widetilde{R}^i)^T g^{i2}$
        2. In this class, we will calculate:
            $\widetilde{R}^i$
        3. The format of $\widetilde{R}^i$ is specific, you can check it in:
            - Zhang L, Han J, Wang H, et al. End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems[J]. Advances in neural information processing systems, 2018, 31.
    
    
    Attributes
    ----------
        1. self.rcut: float
            -
        2. self.rcut_smooth: float
            -
        3. self.dp_feature_pair_an: np.ndarray
            - shape = (num_centers, num_nbrs)
        4. self.dp_feature_pair_d: np.ndarray
            - shape = (num_centers, num_nbrs)
        5. self.dp_feature_pair_rc: np.ndarray
            - shape = (num_centers, num_nbrs, 3)
        6. self.dp_feature_pair_tildeR: np.nadrray
            - shape = (num_centers, num_nbrs, 4)
    '''
    def __init__(
                self,
                structure_neighbors:StructureNeighborsV1,
                center_atomic_number:int,
                nbr_atomic_number:int,
                rcut:float,
                rcut_smooth:float):
        '''
        Parameters
        ----------
            1. structure_neighbors: StructureNeighborsV1
            2. center_atomic_number: int
            3. nbr_atomic_number: int
            4. rcut: float
            5. rcut_smooth: float
        '''
        self.rcut = rcut
        self.rcut_smooth = rcut_smooth
        self.dp_feature_pair_an, self.dp_feature_pair_d, self.dp_feature_pair_rc = \
                    self._get_premise(
                            structure_neighbors=structure_neighbors,
                            center_atomic_number=center_atomic_number,
                            nbr_atomic_number=nbr_atomic_number)
        self.dp_feature_pair_tildeR = self._get_tildeR(rcut=rcut, rcut_smooth=rcut_smooth)
    
    
    def _get_premise(
                    self,
                    structure_neighbors:StructureNeighborsV1,
                    center_atomic_number:int,
                    nbr_atomic_number:int):
        '''
        Description
        -----------
            1. 得到计算 $\widetilde{R}^i$ 所需的基本信息:
                1) 近邻原子的元素种类
                2) 近邻原子距中心原子的距离
                3) 近邻原子距中心原子的相对坐标
            2. 本类针对“中心原子-近邻原子”，即固定了中心原子、近邻原子的元素种类，因此我们将其称为`Pair`
        
        Parameters
        ----------
            1. structure_neighbors: DpFeaturePairPremise
                - 
            2. center_atomic_number: int 
                - 中心原子的原子序数
            3. nbr_atomic_number: int
                - 近邻原子的原子序数 
                
        Return 
        ------
            1. dp_feature_pair_an: np.ndarray, shape = (num_center, max_num_nbrs_real_element)
                - 近邻原子的元素种类
            2. dp_feature_pair_d: np.ndarray, shape = (num_center, max_num_nbrs_real_element)
                - 近邻原子距中心原子的欧式距离 (在笛卡尔坐标系下)
            3. dp_feature_pair_rc: np.ndarray, shape = (num_center, max_num_nbrs_real_element, 3)
                - 近邻原子距中心原子的相对坐标 (在笛卡尔坐标系下)
        '''
        dp_feature_pair_premise = DpFeaturePairPremiseDescriptor.create(
                                    "v1",
                                    structure_neighbors=structure_neighbors)
        dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc = \
                dp_feature_pair_premise.extract_feature_pair(
                    center_atomic_number=center_atomic_number,
                    nbr_atomic_number=nbr_atomic_number)
                
        return dp_feature_pair_an, dp_feature_pair_d, dp_feature_pair_rc
    
    
    def _get_s(self, rcut:float, rcut_smooth:float):
        '''
        Description
        -----------
            1. 由 `self.dp_feature_pair_d` 构建 `dp_feature_pair_s`
                - `s` 是一个分段函数
                - `s` 的具体形式见 Zhang L, Han J, Wang H, et al. End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems[J]. Advances in neural information processing systems, 2018, 31.
            2. 距离大于 `rcut` 的已经在 `DpFeaturePairPremise` 中被设置为 0 了
        
        Parameters
        ----------
            1. rcut: float
                - 近邻原子距中心原子的距离超过 `rcut` 时，不予考虑
            2. rcut_smooth: float 
                - 近邻原子距中心原子的距离超过 `rcut_smooth` 时，计算对应的分段函数形式
                
        Return
        ------
            1. dp_feature_pair_s: np.ndarray, shape=(num_center, max_num_nbrs_real_element)
                - s 是根据 `近邻原子距中心原子的距离` 计算得出的，是一个分段函数形式
            
        '''        
        ### Step 1. 获取 `dp_feature_pair_d_reciprocal` -- $\frac{1}{rji}$
        # (num_center, max_num_nbrs_real_element)
        dp_feature_pair_d_reciprocal_ = np.reciprocal(self.dp_feature_pair_d)
        dp_feature_pair_d_reciprocal = np.where(self.dp_feature_pair_d==0, 0, dp_feature_pair_d_reciprocal_)

        ### Step 2. 把`self.dp_feature_pair_d`全部转换为 rcut_smooth < r < rcut 时的形式
        # (num_center, max_num_nbrs_real_element)
        # Version 2018: dp_feature_pair_d_scaled = dp_feature_pair_d_reciprocal * (1/2) * (np.cos(np.pi*(self.dp_feature_pair_d-rcut_smooth)/(rcut-rcut_smooth)) + 1)
        dp_feature_x = (self.dp_feature_pair_d - rcut_smooth) / (rcut - rcut_smooth)
        dp_feature_pair_d_scaled = dp_feature_pair_d_reciprocal * \
                ( 
                    np.power(dp_feature_x, 3) * (-6 * np.power(dp_feature_x, 2) + 15 * dp_feature_x - 10) + 1
        )
        
        ### Step 3. 根据 Step2. 和 Step3. 的结果筛选 
        # (num_center, max_num_nbrs_real_element)
        dp_feature_pair_s = np.where(
                                (self.dp_feature_pair_d>rcut_smooth) & (self.dp_feature_pair_d<rcut),
                                dp_feature_pair_d_scaled,
                                dp_feature_pair_d_reciprocal)
        #print(dp_feature_pair_s)

        return dp_feature_pair_s
    
    
    def _get_tildeR(self, rcut:float, rcut_smooth:float):
        '''
        Description
        -----------
            1. Get $\widetilde{R}$ in Zhang L, Han J, Wang H, et al. End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems[J]. Advances in neural information processing systems, 2018, 31.
            
        Parameters
        ----------
            1. rcut: float
                - 近邻原子距中心原子的距离超过 `rcut` 时，不予考虑
            2. rcut_smooth: float 
                - 近邻原子距中心原子的距离超过 `rcut_smooth` 时，计算对应的分段函数形式      
            
        Return
        ------
            1. dp_feature_pair_tildeR: np.ndarray, shape=(num_center, max_num_nbrs_real_element, 4)
                - $\widetilde{R}$ in Zhang L, Han J, Wang H, et al. End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems[J]. Advances in neural information processing systems, 2018, 31.
                
        Note
        ----
            1. Haven't use zero-padding.
        '''
        ### Step 1. 调用 `self._get_s()` 得到 `dp_feature_pair_s`
        # shape = (num_center, max_num_nbrs_real_element)
        dp_feature_pair_s = self._get_s(rcut=rcut, rcut_smooth=rcut_smooth)
        # shape = (num_center, max_num_nbrs_real_element, 1)
        dp_feature_pair_s = np.repeat(
                                dp_feature_pair_s[:, :, np.newaxis],
                                1,
                                axis=2)
        
        ### Step 2. 利用 `self.dp_feature_pair_rc` 计算 $\widetilde{R}$ 的后三列
        ### Step 2.1.
        # shape = (num_center, max_num_nbrs_real_element)
        dp_feature_pair_d_reciprocal_ = np.reciprocal(self.dp_feature_pair_d)
        dp_feature_pair_d_reciprocal = np.where(self.dp_feature_pair_d==0, 0, dp_feature_pair_d_reciprocal_)
        
        # shape = (num_center, max_num_nbrs_real_element, 1)
        dp_feature_pair_d_reciprocal = np.repeat(
                                dp_feature_pair_d_reciprocal[:, :, np.newaxis],
                                1,
                                axis=2)
        
        ### Step 2.2.
        # shape = (num_center, max_num_nbrs_real_element, 3)
        tildeR_last3 = dp_feature_pair_s * dp_feature_pair_d_reciprocal * self.dp_feature_pair_rc

        ### Step 3. 合并 `dp_feature_pair` 和 `tildeR_last3`
        # shape = (num_center, max_num_nbrs_real_element, 4)
        dp_feature_pair_tildeR = np.concatenate(
                                        [dp_feature_pair_s, tildeR_last3],
                                        axis=2)
        
        return dp_feature_pair_tildeR
    
    
    def get_tildeR(self, max_num_nbrs:int):
        '''
        Description
        -----------
            1. 按照 `max_num_nbrs`，对 `self.dp_feature_pair_tildeR` 进行 zero-padding

        Return
        ------
            1. dp_feature_pair_tildeR: np.ndarray, shape=(num_center, max_num_nbrs_real_element, 4)
        '''
        tilde_R = np.zeros((
                    self.dp_feature_pair_tildeR.shape[0],
                    max_num_nbrs,
                    4)
        )
        tilde_R[:, :self.dp_feature_pair_tildeR.shape[1], :] = \
                    self.dp_feature_pair_tildeR
        
        return tilde_R
    
    
    def _calc_derivative(self):
        '''
        Description
        -----------
            1. 计算 $\tilde{R_i}$ 相对于 `x, y, z` 的导数
                - $\tilde{R_i} = {
                                s(r_{ij}), 
                                \frac{s(r_{ij}x_{ij})}{r_{ij}}, 
                                \frac{s(r_{ij}y_{ij})}{r_{ij}}, 
                                \frac{s(r_{ij}z_{ij})}{r_{ij}}}$
                - $\tilde{R_i}$.shape = (num_centers, num_nbrs, 4)
                - $\tilde{R_i}$ 的导数.shape = (num_centers, num_nbrs, 4, 3)
        
        Parameters
        ----------
            1. 

        Return
        ------
            1. 
            
        Temp variables
        --------------
            1. uu = \frac{r-r_s}{r_c-r_s}
            2. duu = \frac{uu}{dr} = \frac{1}{r_c-r_s}
            3. vv = s(r) * r
                1) r < r_s          : 1
                2) r_s <= r < r_c   : uu^3 (-6uu^2 + 15uu - 10) + 1
                3) r >= r_c         : 0
            4. dvv = 
                1) r < r_s          : 0
                2) r_s <= r < r_c   : [3uu^2(-6uu^2 + 15uu -10) + uu^3(-12uu+15)] * \frac{1}{r_c-r_s}
                3) r >= r_c         : 0
        '''
        # self.dp_feature_pair_d: 
        #   1. 满足 center_atomic_number, nbr_atomic_number
        #   2. 近邻原子距中心原子的距离
        #   3. 按照由近及远的顺序排列
        #   4. Note: 没有 zero_padding, 仅考虑了实体原子！！！
        #print(self.dp_feature_pair_d)
        
        ### Step 1. 计算 mask_1, mask_2 对应的 vv, dvv
        #       1) uu = \frac{r-r_s}{r_c-r_s}
        #       2) duu = duu/dr = \frac{1}{r_c-r_s}
        #       3) vv = s(r) * r
        #       4) dvv = ds(r)/dr
        
        ### Step 1.1. uu, duu
        # shape = (num_centers, num_nbrs)
        uu = (self.dp_feature_pair_d - self.rcut_smooth) / (self.rcut - self.rcut_smooth)
        duu = 1 / (self.rcut - self.rcut_smooth)
        
        ### Step 1.2. vv, dvv        
        ### Step 1.2.1. mask_1: r < rcut
        # shape = (num_centers, num_nbrs)
        vv_mask_1 = 1
        dvv_mask_1 = 0
        
        ### Step 1.2.2. mask_2: rcut_smooth <= r < rcut
        # shape = (num_centers, num_nbrs)
        vv_mask_2 = np.power(uu, 3) * (-6*np.power(uu, 2) + 15*uu -10) + 1
        dvv_mask_2 = ( 
                    3*np.power(uu, 2) * (-6*np.power(uu,2) + 15*uu - 10) + \
                    np.power(uu, 3) * (-12*uu + 15)
        ) * duu
        
        ### Step 1.2.3. 根据 vv_mask_1, vv_mask_2 计算 vv
        # shape = (num_centers, num_nbrs)
        vv = np.where(
                self.dp_feature_pair_d<self.rcut_smooth,
                vv_mask_1,
                vv_mask_2
        )
        
        ### Step 1.2.4. 根据 dvv_mask_1, dvv_mask_2 计算 dvv
        # shape = (num_centers, num_nbrs)
        dvv = np.where(
                self.dp_feature_pair_d<self.rcut_smooth,
                dvv_mask_1,
                dvv_mask_2
        )        
        
        
        ### Step 2. Calculate the derivative of (1/r * vv)
        # tildeR_deriv.shape = (num_centers, num_nbrs, 4, 3)
        tildeR_deriv = np.zeros(
                shape=(
                    self.dp_feature_pair_d.shape[0],
                    self.dp_feature_pair_d.shape[1],
                    4,
                    3)
        )
        
        '''
        Math
        ----    
        1. Formula 1
            $            
            \frac{d(\frac{1}{r} \cdot vv)}{dx}
            = vv \cdot \frac{d\frac{1}{r}}{dx} + \frac{1}{r} \cdot \frac{dvv}{dx}
            $
            
        2. Formula 2
            $
            \frac{d\frac{1}{r}}{x} 
            = \frac{d\frac{1}{r}}{dr} \cdot \frac{dr}{dx}
            = -\frac{1}{r^2}(-\frac{x}{r}) = \frac{x}{r^3}
            $
            
        3. Formula 3
            $
            \frac{dvv}{dx} = \frac{dvv}{dr} \cdot \frac{dr}{dx} = - dvv \cdot \frac{x}{r}
            $
        '''
        dp_feature_pair_d_reciprocal_ = np.reciprocal(self.dp_feature_pair_d)
        dp_feature_pair_d_reciprocal = np.where(
                            self.dp_feature_pair_d==0,
                            0,
                            dp_feature_pair_d_reciprocal_)
        
        ### Step 2.1. d(1/r * vv) / dx
        tildeR_deriv[:, :, 0, 0] = (
            vv * self.dp_feature_pair_rc[:, :, 0] * np.power(dp_feature_pair_d_reciprocal, 3) - \
            dp_feature_pair_d_reciprocal * dvv * self.dp_feature_pair_rc[:, :, 0] * dp_feature_pair_d_reciprocal
        )
        
        ### Step 2.2. d(1/r * vv) / dy
        tildeR_deriv[:, :, 0, 1] = (
            vv * self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 3) - \
            dp_feature_pair_d_reciprocal * dvv * self.dp_feature_pair_rc[:, :, 1] * dp_feature_pair_d_reciprocal
        )
        
        ### Step 2.3. d(1/r * vv) / dz
        tildeR_deriv[:, :, 0, 2] = (
            vv * self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 3) - \
            dp_feature_pair_d_reciprocal * dvv * self.dp_feature_pair_rc[:, :, 2] * dp_feature_pair_d_reciprocal
        )
        
        
        ### Step 3. Calculate the derivative of (x/r^2 * vv)
        '''
        Math
        ----
        1. Formula 1
            $
            \frac{d\frac{x}{r^2}vv}{dx}
            = vv \cdot \frac{d\frac{x}{r^2}}{dx} + \frac{x}{r^2} \cdot \frac{dvv}{dx}
            $
        2. Formula 2
            $
            \frac{d\frac{x}{r^2}}{dx}
            = vv \cdot (\frac{2x^2}{r^4} - \frac{1}{r^2}) - (\frac{x}{r^2} \cdot dvv \cdot \frac{x}{r})
            $
        3. Formula 3
            $
            \frac{dvv}{dx} = \frac{dvv}{dr} \cdot \frac{dr}{dx} = - dvv \cdot \frac{x}{r}
            $
        '''
        ### Step 3.1. d(x/r * 1/r * vv) / dx -> d(x/r^2 * vv) / dx
        tildeR_deriv[:, :, 1, 0] = (
            vv * (2 * np.power(self.dp_feature_pair_rc[:, :, 0], 2) * np.power(dp_feature_pair_d_reciprocal, 4) - np.power(dp_feature_pair_d_reciprocal, 2)) - \
            self.dp_feature_pair_rc[:, :, 0] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 0] * dp_feature_pair_d_reciprocal
        )
        ### Step 3.2. d(x/r * 1/r * vv) / dy -> d(x/r^2 * vv) / dy
        tildeR_deriv[:, :, 1, 1] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 0] * self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 0] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 1] * dp_feature_pair_d_reciprocal
        )
        ### Step 3.3. d(x/r * 1/r * vv) / dz -> d(x/r^2 * vv) / dz
        tildeR_deriv[:, :, 1, 2] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 0] * self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 0] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 2] * dp_feature_pair_d_reciprocal
        )
                
        
        ### Step 4. Calculate the derivative of (y/r^2 * vv)
        ### Step 4.1. d(y/r^2 * vv) / dx
        tildeR_deriv[:, :, 2, 0] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 0] * self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 0] * dp_feature_pair_d_reciprocal
        )
        ### Step 4.2. d(y/r^2 * vv) / dy
        tildeR_deriv[:, :, 2, 1] = (
            vv * (2 * np.power(self.dp_feature_pair_rc[:, :, 1], 2) * np.power(dp_feature_pair_d_reciprocal, 4) - np.power(dp_feature_pair_d_reciprocal, 2)) - \
            self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 1] * dp_feature_pair_d_reciprocal
        )
        ### Step 4.3. d(y/r^2 * vv) / dz
        tildeR_deriv[:, :, 2, 2] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 1] * self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 1] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 2] * dp_feature_pair_d_reciprocal
        )

        
        ### Step 5. Calculate the derivative of (z/r^2 * vv)
        ### Step 5.1. d(z/r^2 * vv) / dx
        tildeR_deriv[:, :, 3, 0] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 0] * self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 0] * dp_feature_pair_d_reciprocal
        )
        ### Step 5.2. d(z/r^2 * vv) / dy
        tildeR_deriv[:, :, 3, 1] = (
            vv * (2 * self.dp_feature_pair_rc[:, :, 1] * self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 4)) - \
            self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 1] * dp_feature_pair_d_reciprocal
        )   
        ### Step 5.3. d(z/r^2 * vv) / dz
        tildeR_deriv[:, :, 3, 2] = (
            vv * (2 * np.power(self.dp_feature_pair_rc[:, :, 2], 2) * np.power(dp_feature_pair_d_reciprocal, 4) - np.power(dp_feature_pair_d_reciprocal, 2)) - \
            self.dp_feature_pair_rc[:, :, 2] * np.power(dp_feature_pair_d_reciprocal, 2) * dvv * self.dp_feature_pair_rc[:, :, 2] * dp_feature_pair_d_reciprocal
        )
        
        return tildeR_deriv
    

    def calc_derivative(self, max_num_nbrs:int):
        '''
        Description
        -----------
            1. 按照 `max_num_nbrs`，对 `self.dp_feature_pair_tildeR` 的导数进行 zero-padding
        
        Return 
        ------
            1. tildeR_deriv: np.ndarray
                - shape = (num_centers, max_num_nbrs, 4, 3)
        '''
        tildeR_deriv = np.zeros((
            self.dp_feature_pair_tildeR.shape[0],
            max_num_nbrs,
            4, 3)
        )
        tildeR_deriv[:, :self.dp_feature_pair_tildeR.shape[1], :, :] = \
                self._calc_derivative()
                
        return tildeR_deriv
