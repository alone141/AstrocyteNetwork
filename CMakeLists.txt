cmake_minimum_required(VERSION 3.20)

# Set the project name and version

project(AstrocyteNetwork
    VERSION 0.8.0
    DESCRIPTION "An archaic neural network project that completely trains on compile time"
    HOMEPAGE_URL "https://github.com/alone141/AstrocyteNetwork"
    LANGUAGES CXX
)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compile-time training runs inside the compiler, and every compiler gives up on a constant
# expression after a fixed amount of work. Each compiler has its own flag to raise that limit.
# Call this for every target that trains a network at compile time.
function(astrocyte_enable_compile_time_training target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /constexpr:steps100000000 /Zm2000)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -fconstexpr-ops-limit=4294967295 -fconstexpr-loop-limit=1000000000)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -fconstexpr-steps=2147483647)
    endif()
endfunction()

function(astrocyte_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Add an option to enable or disable tests
option(ENABLE_TEST "Enable testing" OFF)

if(ENABLE_TEST)
  enable_testing()

  # Use an installed GoogleTest if there is one, otherwise download it
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  include(GoogleTest)
endif()

add_subdirectory(Utility)
add_subdirectory(Perceptron)
add_subdirectory(AstrocyteNetwork)
add_subdirectory(Main)
