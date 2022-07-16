cmake_minimum_required(VERSION 3.17)
project(cpp_project)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(OpenCV_DIR "D:\\Program Files (x86)\\opencv\\m_build\\install\\x64\\mingw\\lib")
find_package(OpenCV 4.6.0 REQUIRED)


add_executable(cpp_project main.cpp MyMatrix.h demo.h)

target_link_libraries(cpp_project ${OpenCV_LIBS})