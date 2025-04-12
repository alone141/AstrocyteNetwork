add_library(AstrocyteNetwork INTERFACE)
target_compile_features(AstrocyteNetwork INTERFACE cxx_std_23)

target_include_directories(AstrocyteNetwork
    PUBLIC INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(AstrocyteNetwork
    PUBLIC
        "include/AstrocyteNetwork.h"
        "include/IAstrocyteNetwork.h"
)

target_link_libraries(AstrocyteNetwork
    PUBLIC INTERFACE Perceptron
)

if(ENABLE_TEST)
    add_executable(AstrocyteNetworkTest ${CMAKE_CURRENT_SOURCE_DIR}/test/src/AstrocyteNetworkFunctionalTest.cpp)
    target_link_libraries(AstrocyteNetworkTest AstrocyteNetwork GTest::gtest_main)
    target_include_directories(AstrocyteNetworkTest
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
)
    include(GoogleTest)
    gtest_discover_tests(AstrocyteNetworkTest)
endif()