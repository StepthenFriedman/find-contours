set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.10)

project(find-contours CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CUDA_VERSION 11.5)

find_package(CUDA 11 REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/run.cu)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)
