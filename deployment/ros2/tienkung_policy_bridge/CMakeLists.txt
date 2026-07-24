cmake_minimum_required(VERSION 3.8)
project(tienkung_policy_bridge)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(policy_bridge_node src/policy_bridge_node.cpp)
target_compile_features(policy_bridge_node PUBLIC cxx_std_17)
target_include_directories(policy_bridge_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(policy_bridge_node rclcpp sensor_msgs std_msgs)

install(TARGETS policy_bridge_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_safety_guard test/test_safety_guard.cpp)
  target_compile_features(test_safety_guard PUBLIC cxx_std_17)
  target_include_directories(test_safety_guard PRIVATE include)
  add_test(NAME safety_guard COMMAND test_safety_guard)
endif()

ament_package()
