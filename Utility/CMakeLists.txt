add_library(Utility INTERFACE)
target_compile_features(Utility INTERFACE cxx_std_23)

target_include_directories(Utility
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Listed so IDEs show the headers; header-only libraries have nothing to compile
target_sources(Utility
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h
)

if(ENABLE_TEST)
    add_executable(UtilityUnitTest ${CMAKE_CURRENT_SOURCE_DIR}/test/src/UtilityUnitTest.cpp)
    target_link_libraries(UtilityUnitTest PRIVATE Utility GTest::gtest)
    astrocyte_enable_compile_time_training(UtilityUnitTest)
    astrocyte_enable_warnings(UtilityUnitTest)

    gtest_discover_tests(UtilityUnitTest)
endif()
