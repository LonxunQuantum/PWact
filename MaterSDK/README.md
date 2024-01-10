# Material Software Development Kit (MaterSDK)
A python library for computational materials science.


# 1. Installation
## 1.1. Online
```shell
$ git clone git@github.com:lhycms/MaterSDK.git
$ cd matersdk
$ pip install .
```

## 1.2. Offline
1. You can download a python interpreter containing `matersdk` from https://www.jianguoyun.com/p/DfhQFx8Q_qS-CxifgfwEIAA.matersdk.egg-info

# 2. Integrate MaterSDK with `lammps` as third-party libraries
```shell
# Step 1.
$ mkdir <your_path_lammps>/lammps/src/MATERSDK
$ cd <your_path_lammps>/lammps/src/MATERSDK

# Step 2.
$ cp <your_path_to_matersdk>/matersdk/source ./.
$ cp <your_path_to_matersdk>/matersdk/cmake ./.
$ ls .
source cmake LICENSE
$ vim <your_path_to_lammps>/src/MATERSDK/source/CMakeLiest.txt
### Modify the path of MATERSDK
add_subdirectory("<your_path_to_lammps>/src/MATERSDK/source/ext/googletest-main")
$ cd vim <your_path_to_lammps>/src/MATERSDK/source/build; cmake -DBUILD_TEST=1 -DTORCH_OP=1 ..; make

# Step 3. 
$ vim <your_path_to_lammps>/cmake/CMakeLists.txt
### Add content below to CMakeLists.txt
include("<your_path_to_lammps>/src/MATERSDK/cmake/matersdk.cmake")   ### matersdk
include_directories(${MATESDK_INCLUDE_DIRS}) ### matesdk

# Link libraries of matersdk to lammps.so/lammps.a
add_library(lammps ${ALL_SOURCES})  ### matersdk
target_link_libraries(lammps PUBLIC ${MATERSDK_LIBRARIES})  ### matersdk

# Step 4. 
$ cd <your_path_to_lammps>/build
$ cmake ../cmake 
-- +++ MATERSDK_DIR : /data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK
-- +++ MATERSDK_INCLUDE_DIRS : /data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK/source/include;/data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK/source/core/include;/data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK/source/matersdk/io/publicLayer/include;/data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK/source/matersdk/feature/deepmd/include
-- +++ MATERSDK_LIBRARIES : /data/home/liuhanyu/hyliu/code1/lammps_gpu/src/MATERSDK/source/build/lib/matersdk/feature/deepmd/se4pw_op
```