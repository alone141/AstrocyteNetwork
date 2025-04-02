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
    template <typename BaseType, typename ExpType> 
    requires std::is_arithmetic_v<BaseType> && std::is_integral_v<ExpType>
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
}

#endif // UTILITY_H