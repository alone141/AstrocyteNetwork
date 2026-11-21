# Add an executable
add_executable(Main main.cpp)

target_link_libraries(Main PRIVATE AstrocyteNetwork)
astrocyte_enable_compile_time_training(Main)
astrocyte_enable_warnings(Main)
