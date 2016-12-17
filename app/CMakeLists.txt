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

#find_package(freenect2)

# add_executable(kinect_demo
#   kinect_demo.cpp)

# set(FREENECT2_INCLUDE_DIRS "/usr/local/include/libfreenect2/include")
# set(FREENECT2_LIBRARY "/usr/local/lib/libfreenect2.so")

# target_link_libraries(kinect_demo
#   RandomForest
#   ${OpenCV_LIBS}
#   ${FREENECT2_LIBRARY})
