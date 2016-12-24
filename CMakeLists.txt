cmake_minimum_required(VERSION 2.8.9)
project(rfclassifier)

add_definitions(-std=c++11)

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "-Ofast")

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(src)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
