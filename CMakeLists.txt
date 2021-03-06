cmake_minimum_required(VERSION 2.8.9)
project(rfclassifier)

add_definitions(-std=c++1y)

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "-Ofast -march=native")

add_definitions(-DDATA_FOLDER="${CMAKE_SOURCE_DIR}/data")

option(WITH_KINECT "Compile with Kinect support" OFF)
set(WITH_OPENCL "GPU" CACHE STRING "Type of OpenCL backend [GPU|CPU|OFF]")
set_property(CACHE WITH_OPENCL PROPERTY STRINGS "OFF" "GPU" "CPU")

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (${WITH_OPENCL} MATCHES "GPU")
  #include(cmake/clblas.cmake)
  add_definitions(-DUSING_OPENCL=1)
  add_definitions(-DOPENCL_DEVICE=0)
  find_package(OpenCL 1.2 REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
#  include_directories(${CLBLAS_INCLUDE_DIR})
elseif(${WITH_OPENCL} MATCHES "CPU")
  #include(cmake/clblas.cmake)
  add_definitions(-DUSING_OPENCL=1)
  add_definitions(-DOPENCL_DEVICE=1)
  find_package(OpenCL 1.2 REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
#  include_directories(${CLBLAS_INCLUDE_DIR})
else()
  add_definitions(-DUSING_OPENCL=0)
  add_definitions(-DOPENCL_DEVICE=-1)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(byfron-utils/src)
include_directories(ska_sort)

include_directories(src)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
