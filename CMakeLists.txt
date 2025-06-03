cmake_minimum_required(VERSION 3.15)
project(SHIFT_FMT) 
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

file(GLOB SOURCES src/*.cpp)
add_subdirectory(thirdparty/glog)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBRARIES}
    PkgConfig::FFTW
    glog::glog 
)
