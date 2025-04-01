cmake_minimum_required(VERSION 3.16)

# Set the project name and version
project(AstrocyteNetwork VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(Main)
add_subdirectory(Perceptron)
add_subdirectory(Utility)