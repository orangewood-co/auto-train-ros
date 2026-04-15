cmake_minimum_required(VERSION 3.8)
project(auto_train_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install the source Python package (existing modules)
ament_python_install_package(source)

# Install executable scripts
install(PROGRAMS
  scripts/autotrain_action_server.py
  scripts/autotrain_action_client.py
  DESTINATION lib/${PROJECT_NAME}
)

# Generate action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/AutoTrain.action"
  DEPENDENCIES std_msgs action_msgs
)

ament_export_dependencies(
  rclpy
  rosidl_default_runtime
)

ament_package()
