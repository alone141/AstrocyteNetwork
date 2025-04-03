#ifndef UTILITY_H
#define UTILITY_H
#include <concepts>
#include <type_traits>
#include <Config.h>
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

#endif // UTILITY_H