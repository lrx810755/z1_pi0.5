cmake_minimum_required(VERSION 3.8)
project(z1_examples)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake        REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy              REQUIRED)

install(
    PROGRAMS
    ${PROJECT_NAME}/waypoint_test.py
    ${PROJECT_NAME}/red_target_handeye_approach.py
    ${PROJECT_NAME}/red_target_gui.py
    ${PROJECT_NAME}/red_ball_camera_moveit.py
    ${PROJECT_NAME}/camera_info_fix.py
    DESTINATION 
    lib/${PROJECT_NAME}
)

install(
    DIRECTORY launch 
    DESTINATION share/${PROJECT_NAME}
)

install(
    FILES requirements-yolo.txt
    DESTINATION share/${PROJECT_NAME}
)

install(
    FILES models/yolo11n.pt
    DESTINATION share/${PROJECT_NAME}/models
)

ament_python_install_package(${PROJECT_NAME})
ament_package()
