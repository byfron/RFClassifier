add_executable(train
  train.cpp)

target_link_libraries(train
  Frame
  RandomForest)

add_executable(predict
  predict.cpp)

target_link_libraries(predict
  Frame
  RandomForest)

find_package(freenect2 REQUIRED)

add_executable(kinect_demo
   kinect_demo.cpp)

target_link_libraries(kinect_demo
   RandomForest
   ${OpenCV_LIBS}
   ${freenect2_LIBRARY})
