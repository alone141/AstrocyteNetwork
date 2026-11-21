#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

namespace config
{
    //Number of iteration steps for mathematical computations
    static constexpr unsigned int MATH_ITERATION_DEPTH = 30;

    //Seed for the initial weights. Fixed so that every build trains the exact same network.
    //Change it if a network gets stuck in a bad local minimum.
    static constexpr std::uint64_t RANDOM_SEED = 42;
}


#endif // CONFIG_H
