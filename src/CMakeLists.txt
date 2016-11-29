add_library(RandomForest
  RandomForest.hpp
  RandomForest.cpp)

target_link_libraries(RandomForest
    ${OpenCV_LIBS})
