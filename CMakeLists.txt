cmake_minimum_required(VERSION 3.30)
project(Project)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 20)
add_executable(Project main.cpp
        proj.cpp
        proj.h)
target_link_libraries(Project ${OpenCV_LIBS})
