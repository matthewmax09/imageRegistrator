cmake_minimum_required(VERSION 3.15)
project(SHIFT_FMT) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(BUILD_TESTING OFF)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

find_library(
    FFTW_DOUBLE_THREADS_LIB
    NAMES "fftw3_omp"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
)
          
if (FFTW_DOUBLE_THREADS_LIB)
    set(FFTW_DOUBLE_THREADS_LIB_FOUND TRUE)
    set(FFTW_LIBRARIES ${FFTW_LIBRARIES} ${FFTW_DOUBLE_THREADS_LIB})
    add_library(FFTW::DoubleThreads INTERFACE IMPORTED)
    set_target_properties(FFTW::DoubleThreads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW_DOUBLE_THREADS_LIB}"
    )
else()
    set(FFTW_DOUBLE_THREADS_LIB_FOUND FALSE)
endif()

file(GLOB SOURCES src/*.cpp)
add_subdirectory(thirdparty/glog)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBRARIES}
    PkgConfig::FFTW
    ${FFTW_DOUBLE_THREADS_LIB}
    glog::glog
    OpenMP::OpenMP_CXX 
)
