add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cpp)
add_library(test_main gtest_main.cc gtest)

file(GLOB test_srcs test_*.cpp)

set(the_target test.testbin)
set(test_args --gtest_shuffle)

add_executable(${the_target} EXCLUDE_FROM_ALL ${test_srcs})
target_link_libraries(${the_target} test_main gtest)

add_custom_target(test COMMAND ${the_target} ${test_args}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
