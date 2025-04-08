add_library(Perceptron INTERFACE)
target_compile_features(Perceptron INTERFACE cxx_std_23)

target_include_directories(Perceptron
    PUBLIC INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(Perceptron
    PUBLIC
        "include/Perceptron.h"
        "include/IPerceptron.h"
        "include/ActivationFunctionEnum.h"
)

target_link_libraries(Perceptron
    PUBLIC INTERFACE Utility
)