cmake_minimum_required(VERSION 2.8.9)
project(rfclassifier)

add_definitions(-std=c++11)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(src)

add_subdirectory(src)
add_subdirectory(app)
#add_subdirectory(test)
