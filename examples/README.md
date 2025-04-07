# 案例说明
## 一、pwmat
使用 PWmat 做主动学习，使用的环境为Mcloud已安装环境。

### si_pwmat/init_bulk
step1. 使用 PWmat 做 relax；
step2. 对结构使用 PWmat 运行 AIMD，轨迹中的结构用于后续主动学习

### si_pwmat/run_iter
在主动学习过程中，对通过多模型偏差筛选出的结构，使用 PWmat 推理结构的能量和受力

## 二、si_pwmat_gaussian
使用 PWmat (gaussion 基组) 做主动学习，使用的环境为Mcloud已安装环境。

### si_pwmat_gaussian/init_bulk
step1. 使用 PWmat (gaussion 基组) 做 relax；
step2. 对结构使用 PWmat (gaussion 基组) 运行 AIMD，轨迹中的结构用于后续主动学习

### si_pwmat_gaussian/run_iter
在主动学习过程中，对通过多模型偏差筛选出的结构，使用 PWmat  (gaussion 基组) 推理结构的能量和受力

## 大模型标注和Direct采样
`注意，案例中提供的 sevennet_md.py 使用 ASE 运行 MD。 ASE 的 NPT 模块要求 模拟盒子必须是上三角矩阵（即 cell[i][j] = 0 当 i > j）。` 因此案例中未做 驰豫，对结构的微扰设置"cell_pert_fraction":0，避免绕动晶格。

## 三、si_pwmatgaussion_bigmodel_direct
使用 PWmat（gaussion）、大模型做标注（计算能量、受力）、大模型做分子动力学习、direct 采样

### si_pwmatgaussion_bigmodel_direct/init_bulk_bigmodel
step1. 使用 PWmat 做relax；
step2. 对结构调用大模型（seventnet）做分子动力学；
step3. 对分子动力学得到的轨迹做 direct 采样，去掉轨迹中相似的结构，筛选出的结构用于后续主动学习

### si_pwmatgaussion_bigmodel_direct/init_bulk_pwmat
step1. 使用 PWmat 做relax；
step2. 对结构调用大模型（seventnet）做分子动力学；
step3. 对分子动力学得到的轨迹做 direct 采样，去掉轨迹中相似的结构；
step4. 对筛选出的结构调用 PWmat 做自洽计算（标注结构的能量和受力），之后用于后续主动学习

### si_pwmatgaussion_bigmodel_direct/run_iter_direct_bigmodel
在主动学习过程中，对通过多模型偏差筛选出的结构，调用direct采样去掉重复结构；使用大模型（eqv2）推理结构的能量和受力

### si_pwmatgaussion_bigmodel_direct/run_iter_direct_pwmat
在主动学习过程中，对通过多模型偏差筛选出的结构，调用direct采样去掉重复结构；使用 PWmat 推理结构的能量和受力

### si_pwmatgaussion_bigmodel_direct/run_iter_bigmodel
在主动学习过程中，对通过多模型偏差筛选出的结构，调用direct采样去掉重复结构；使用大模型（eqv2）推理结构的能量和受力

## 四、si_pwmat_bigmodel_direct 

与 si_pwmatgaussion_bigmodel_direct 功能相同，区别是si_pwmatgaussion_bigmodel_direct使用的是gaussian基组，而 si_pwmat_bigmodel_direct 使用 PBE。

## 五、si_vasp
使用 vasp 做主动学习，使用的环境为Mcloud已安装环境。

### si_vasp/init_bulk
step1. 使用 vasp 做 relax；
step2. 对结构使用 vasp 运行 AIMD，轨迹中的结构用于后续主动学习

### si_vasp/run_iter
在主动学习过程中，对通过多模型偏差筛选出的结构，使用 vasp 推理结构的能量和受力

## 六、si_cp2k
使用 cp2k 做主动学习，使用的环境为Mcloud已安装环境。

### si_cp2k/init_bulk
step1. 使用 cp2k 做 relax；
step2. 对结构使用 cp2k 运行 AIMD，轨迹中的结构用于后续主动学习

### si_cp2k/run_iter
在主动学习过程中，对通过多模型偏差筛选出的结构，使用 cp2k 推理结构的能量和受力
