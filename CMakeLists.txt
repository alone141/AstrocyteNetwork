cmake_minimum_required(VERSION 3.16)

# Set the project name and version

project(AstrocyteNetwork
    VERSION 0.7.0
    DESCRIPTION "An archaic neural network project that completely trains on compile time"
    HOMEPAGE_URL "https://github.com/alone141/AstrocyteNetwork"
    LANGUAGES CXX
)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add an option to enable or disable tests
option(ENABLE_TEST "Enable testing" ON)

if(ENABLE_TEST)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(Utility)
add_subdirectory(Perceptron)
add_subdirectory(AstrocyteNetwork)
add_subdirectory(Main)

