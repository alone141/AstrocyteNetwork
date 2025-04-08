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