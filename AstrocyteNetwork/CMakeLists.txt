add_library(AstrocyteNetwork INTERFACE)
target_compile_features(AstrocyteNetwork INTERFACE cxx_std_23)

target_include_directories(AstrocyteNetwork
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Listed so IDEs show the headers; header-only libraries have nothing to compile
target_sources(AstrocyteNetwork
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/AstrocyteNetwork.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/IAstrocyteNetwork.h
)

target_link_libraries(AstrocyteNetwork
    INTERFACE Perceptron
)

if(ENABLE_TEST)
    add_executable(AstrocyteNetworkUnitTest ${CMAKE_CURRENT_SOURCE_DIR}/test/src/AstrocyteNetworkUnitTest.cpp)
    target_link_libraries(AstrocyteNetworkUnitTest PRIVATE AstrocyteNetwork GTest::gtest)
    astrocyte_enable_compile_time_training(AstrocyteNetworkUnitTest)
    astrocyte_enable_warnings(AstrocyteNetworkUnitTest)

    gtest_discover_tests(AstrocyteNetworkUnitTest)
endif()
