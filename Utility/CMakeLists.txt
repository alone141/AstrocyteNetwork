add_library(Utility)

target_include_directories(Utility
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
) 

target_sources(Utility
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Utility.cpp
)

