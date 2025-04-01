add_library(Perceptron)

target_include_directories(Perceptron
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(Perceptron
    PRIVATE
        "src/Perceptron.cpp"
    PUBLIC
        "include/Perceptron.h"
        "include/IPerceptron.h"
        "include/ActivationFunctionEnum.h"
)

target_link_libraries(Perceptron
    PUBLIC
        Utility
)