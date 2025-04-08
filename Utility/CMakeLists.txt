add_library(Utility INTERFACE)
target_compile_features(Utility INTERFACE cxx_std_23)

target_include_directories(Utility
    PUBLIC INTERFACE
        "${CMAKE_CURRENT_LIST_DIR}/include"
) 

target_sources(Utility
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h
)

if(ENABLE_TEST)
    add_executable(UtilityUnitTest ${CMAKE_CURRENT_SOURCE_DIR}/test/src/UtilityUnitTest.cpp)
    target_link_libraries(UtilityUnitTest Utility GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(UtilityUnitTest)
endif()