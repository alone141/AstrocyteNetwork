
# Add an executable
add_executable(Main main.cpp)

target_link_libraries(Main PUBLIC AstrocyteNetwork Perceptron Utility)
target_compile_options(Main PRIVATE /constexpr:steps100000000)
target_compile_options(Main PRIVATE /Zm2000)
