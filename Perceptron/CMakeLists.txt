add_library(Perceptron INTERFACE)
target_compile_features(Perceptron INTERFACE cxx_std_23)

target_include_directories(Perceptron
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Listed so IDEs show the headers; header-only libraries have nothing to compile
target_sources(Perceptron
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Perceptron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/IPerceptron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ActivationFunctionEnum.h
)

target_link_libraries(Perceptron
    INTERFACE Utility
)
