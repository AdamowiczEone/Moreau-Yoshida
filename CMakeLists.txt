cmake_minimum_required(VERSION 3.10)
project(Moreau-Yoshida-image-denoise)

add_subdirectory(include/matplotplusplus)
include_directories(include/Eigen/Eigen)
#include_directories(include/matplotplusplus/source/matplot)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Требуем поддержку C++17

find_package(Eigen3 REQUIRED)
add_executable(mainfile "src/main.cpp")
add_executable(eigen_examples "src/eigen_examples.cpp")
add_executable(moreau "src/Moreau.cpp")
target_link_libraries(moreau PUBLIC matplot)
add_executable(plots "src/plot_examples.cpp")
target_link_libraries(plots PUBLIC matplot)