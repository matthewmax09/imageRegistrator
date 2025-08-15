cmake_minimum_required(VERSION 3.15)
project(SHIFT_FMT) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(BUILD_TESTING OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
# Adopt https://github.com/egpbos/findFFTW to simplify CMakeLists
find_package(FFTW REQUIRED COMPONENTS FLOAT_OPENMP_LIB)

file(GLOB SOURCES src/*.cpp)
add_subdirectory(thirdparty/glog)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBRARIES}
    ${FFTW_LIBRARIES}
    glog::glog
    OpenMP::OpenMP_CXX 
)
