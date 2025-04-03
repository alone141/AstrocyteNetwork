add_library(AstrocyteNetwork)

target_include_directories(AstrocyteNetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(AstrocyteNetwork
    PRIVATE
        "src/AstrocyteNetwork.cpp"
    PUBLIC
        "include/AstrocyteNetwork.h"
        "include/IAstrocyteNetwork.h"
)

target_link_libraries(AstrocyteNetwork
    PUBLIC
        Perceptron
)