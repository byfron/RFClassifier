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
