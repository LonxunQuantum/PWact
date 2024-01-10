### Usage: Please `include(matersdk.cmake)` in you CMakeList.txt

# 1. Basic
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(matersdk)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# 2. Include

# 3. Set variable
get_filename_component(MATERSDK_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)


## 3.1. MATERSDK_INCLUDE_DIRS
set(MATERSDK_INCLUDE_DIRS)
list(APPEND MATERSDK_INCLUDE_DIRS ${MATERSDK_DIR}/source/include;)
list(APPEND MATERSDK_INCLUDE_DIRS ${MATERSDK_DIR}/source/core/include;)
list(APPEND MATERSDK_INCLUDE_DIRS ${MATERSDK_DIR}/source/matersdk/io/publicLayer/include;)
list(APPEND MATERSDK_INCLUDE_DIRS ${MATERSDK_DIR}/source/matersdk/feature/deepmd/include;)

## 3.2. MATERSDK_LIBRARIES
set(MATERSDK_LIBRARIES)
list(APPEND MATERSDK_LIBRARIES ${MATERSDK_DIR}/source/build/lib/matersdk/feature/deepmd/libse4pw_op.so;)

message(STATUS "+++ MATERSDK_DIR : ${MATERSDK_DIR}")
message(STATUS "+++ MATERSDK_INCLUDE_DIRS : ${MATERSDK_INCLUDE_DIRS}")
message(STATUS "+++ MATERSDK_LIBRARIES : ${MATERSDK_LIBRARIES}")
