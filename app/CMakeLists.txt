add_executable(train
  train.cpp)

target_link_libraries(train
  ${OpenCV_LIBS}
  Frame
  RandomForest)

add_executable(train_multistage
  train_multistage.cpp)

target_link_libraries(train_multistage
  ${OpenCV_LIBS}
  Frame
  RandomForest)


add_executable(predict
  predict.cpp)

target_link_libraries(predict
  ${OpenCV_LIBS}
  Frame
  RandomForest)

if (${WITH_KINECT})
find_package(freenect2 REQUIRED)

add_executable(kinect_demo
   kinect_demo.cpp)

target_link_libraries(kinect_demo
   RandomForest
   ${OpenCV_LIBS}
   ${freenect2_LIBRARY})
endif()
