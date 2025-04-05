#ifndef UTILITY_H
#define UTILITY_H
#include <concepts>
#include <type_traits>
#include <Config.h>
#include <cstdint>
#include <limits>

// Utility.h - Header file for utility functions and classes
namespace utility
{
    constexpr float factorial(int x)
    {
        if(x < 0)
            return 0; 
        else if(x == 0 || x == 1){
            return 1; 
        }
        else{
            return x * factorial(x - 1); 
        }
    
    }

    //pow function for where exponent is an integer
    template <typename BaseType, typename ExpType> 
    requires std::is_arithmetic_v<BaseType> and std::is_integral_v<ExpType>
    constexpr float pow(BaseType base, ExpType exponent){
        if(exponent < 0)
            return 0; 
        else if(exponent == 0){
            return 1; 
        }
        else{
            float result = base;
            for(int i = 1; i < exponent; i++){
                result *= base; 
            }
            return result;
        }
    }
    
    constexpr float exp(float exponent, int iterationDepth = config::MATH_ITERATION_DEPTH){
        float result = 0;
        for(int i =0; i < iterationDepth; i++){
            result += utility::pow(exponent, i)/utility::factorial(i);
        }
        return result;
    }

    constexpr float ln(float x, int iterationDepth = config::MATH_ITERATION_DEPTH){
        float result = 0;
        for(int i = 0; i < iterationDepth; i++){
            result += (1.0/(2.0*i + 1.0))*utility::pow((x-1)/(x+1),2*i +1);
        }
        return 2*result;
    }

    //pow function for where exponent is an integer
    template <typename BaseType, typename ExpType> 
    requires std::is_arithmetic_v<BaseType> and not(std::is_integral_v<ExpType>)
    constexpr float pow(BaseType base, ExpType exponent){
        return utility::exp(exponent * utility::ln(base));
    }
}

namespace utility::af{
    constexpr float Sigmoid(float x) {
        return 1.0f / (1.0f + utility::exp(-x));
    }

    constexpr float Relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    constexpr float Tanh(float x) {
        return (utility::exp(x) - utility::exp(-x)) / (utility::exp(x) + utility::exp(-x));
    }

    constexpr float Linear(float x) {
        return x;
    }

    constexpr float LeakyRelu(float x, float alpha = 0.01f) {
        return x > 0.0f ? x : alpha * x;
    }
}

//Thanks to Jason Turner https://www.youtube.com/watch?v=rpn_5Mrrxf8
namespace utility::random {
    constexpr auto seed()
    {
        std::uint64_t shifted = 0;

        for (const auto c : __TIME__)
        {
            shifted <<= 8;
            shifted |= c;
        }

        return shifted;
    }

    struct PCG
    {
        struct pcg32_random_t { std::uint64_t state = 0;  std::uint64_t inc = seed(); };
        pcg32_random_t rng;
        typedef std::uint32_t result_type;

        constexpr result_type operator()()
        {
            return pcg32_random_r();
        }

        static result_type constexpr min()
        {
            return std::numeric_limits<result_type>::min();
        }

        static result_type constexpr max()
        {
            return std::numeric_limits<result_type>::min();
        }

    private:
        constexpr std::uint32_t pcg32_random_r()
        {
            std::uint64_t oldstate = rng.state;
            // Advance internal state
            rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1);
            // Calculate output function (XSH RR), uses old state for max ILP
            std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            std::uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

    };

    constexpr auto GetRandomUniform(int count)
    {
        PCG pcg;
        while (count > 0) {
            pcg();
            --count;
        }

        return static_cast<float>(pcg()) / INT_MAX - 1.0f;
    }
 
/*     constexpr auto GetRandomUniform(int count)
    {
        auto random = get_random(count);
        return random / PCG::max() * 2.0f - 1.0f;
    } */
}

#endif // UTILITY_H