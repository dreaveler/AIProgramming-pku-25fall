cmake_minimum_required(VERSION 3.20)
project(ProgrammingInAI LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


add_executable(hw3
    testHW3.cu
    function.cu
    tensor.cu
    utils.cu
)

# --- 将修正后的编码设置代码添加到这里 ---
if(MSVC)
    target_compile_options(hw3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/source-charset:utf-8","/execution-charset:utf-8">)
endif()

# 头文件在工程根目录
target_include_directories(hw3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 链接 CUDA 运行时
target_link_libraries(hw3 PRIVATE 
CUDA::cudart
CUDA::cublas
CUDA::curand
)

# Debug 下启用设备端调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hw3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 多 .cu 文件需要可分离编译/设备链接
set_target_properties(hw3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native 
)