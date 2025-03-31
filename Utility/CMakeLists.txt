add_library(Utility)

target_sources(Utility
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility.h
)

target_include_directories(Utility
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
)