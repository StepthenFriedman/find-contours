set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.10)

project(find-contours CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CUDA_VERSION 11.5)

find_package(CUDA 11 REQUIRED)

find_package(OpenCV 4 REQUIRED)

find_package(yaml-cpp)

add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/variance.cu)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${YAML_CPP_INCLUDE_DIR})

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

target_link_libraries(${CMAKE_PROJECT_NAME} ${YAML_CPP_LIBRARIES})
