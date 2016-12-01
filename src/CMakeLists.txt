add_library(Frame
  Frame.hpp
  Frame.cpp
  )

target_link_libraries(Frame
  ${OpenCV_LIBS})

add_library(RandomGenerator
  RandomGenerator.hpp
  RandomGenerator.cpp)

add_library(RandomForest
  RandomForest.hpp
  RandomForest.cpp)

target_link_libraries(RandomForest
  Frame
  RandomGenerator
  ${OpenCV_LIBS})
