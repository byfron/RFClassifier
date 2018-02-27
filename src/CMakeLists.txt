add_definitions(-DOPENCL_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernels/")

add_library(Frame
  Frame.hpp
  Frame.cpp
  )

add_library(Settings
  common.hpp
  Settings.cpp
  )

target_link_libraries(Frame
  ${OpenCV_LIBS})

add_library(Node
  Node.hpp
  Node.cpp)

add_library(RandomGenerator
  RandomGenerator.hpp
  RandomGenerator.cpp)

add_library(RandomForest
  RandomForest.hpp
  RandomForest.cpp)

target_link_libraries(RandomForest
  Frame
  Node
  Settings
  RandomGenerator  
  ${OpenCV_LIBS})

if (${WITH_OPENCL})
target_link_libraries(RandomForest
  OpenCLManager)

add_library(OpenCLManager
  cl_utils.hpp
  cl_utils.cpp
  OpenCLManager.cpp
  OpenCLManager.hpp
  OpenCLKernel.hpp
  OpenCLKernel.cpp
  )

target_link_libraries(OpenCLManager
  ${OpenCL_LIBRARIES}
  )
endif()
