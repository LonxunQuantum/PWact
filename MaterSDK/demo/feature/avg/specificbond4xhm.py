import numpy as np
from scipy.fft import fft
from matersdk.feature.avg.avgbond import AvgBond, PairBond
from matersdk.io.pwmat.output.movement import Movement


### Part I. Custom parameters
#movement_path = "/data/home/liuhanyu/hyliu/pwmat_demo/xhm/MOVEMENT"
#atom1_index = 0
#atom2_index = 49
movement_path = "/data/home/liuhanyu/hyliu/code/mlff/test/demo2/PWdata/data1/MOVEMENT"
atom1_idx = 0
atom2_idx = 49
tot_time = 1000 # AIMD 时长 -- unit: fs
#num_steps = 42001   # 结构数目
num_steps = 550
times_lst = np.linspace(0, tot_time, num_steps)
interval = times_lst[1] - times_lst[0]

# 文件存储路径
bond_txt_path = "./bond.txt"
bondfft_txt_path = "./bondfft.txt"


### Part II. 初始化 MOVEMENT 对象
movement = Movement(movement_path=movement_path)
#dsys = DpLabeledSystem.from_trajectory_s(movement, rcut=rcut)
#print(dsys)


### Part III. 计算某对原子的键长
pairbond = PairBond(
                movement_path=movement_path,
                atom1_idx=atom1_idx,
                atom2_idx=atom2_idx)
print(f"\nCalculating the {atom1_idx}-{atom2_idx} bond length for all frames in MOVEMENT...")
frame_pair_bonds_lst = pairbond.get_frames_pair_bond()
xs_array = np.array(times_lst).reshape(-1, 1)
ys_array = np.array(frame_pair_bonds_lst).reshape(-1, 1)
xys_array = np.concatenate([xs_array, ys_array], axis=1)
np.savetxt(fname=bond_txt_path, X=xys_array)


### Part IV. FFT
fft_results = fft(frame_pair_bonds_lst)
magnitudes = np.abs(fft_results)
sampling_rate = 1 / (interval * 10E-15)
frequencies = np.fft.fftfreq(len(fft_results), d=1/sampling_rate)
frequencies_thz = frequencies / 10E12

xs_array = np.array(frequencies_thz).reshape(-1, 1)
ys_array = np.array(magnitudes).reshape(-1, 1)
xys_array = np.concatenate([xs_array, ys_array], axis=1)
np.savetxt(fname=bondfft_txt_path, X=xys_array)