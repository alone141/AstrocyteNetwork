#ifndef CONFIG_H
#define CONFIG_H

namespace config
{
    static constexpr unsigned int HIDDEN_LAYERS = 1;
    static constexpr unsigned int INPUT_NEURONS = 3;
    static constexpr unsigned int OUTPUT_NEURONS = 1;

    //Number of iteration steps for mathematical computations
    static constexpr unsigned int MATH_ITERATION_DEPTH = 100; 
}


#endif // CONFIG_H