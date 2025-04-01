#ifndef UTILITY_H
#define UTILITY_H
#include <concepts>
#include <type_traits>
// Utility.h - Header file for utility functions and classes
namespace utility
{
    constexpr int factorial(int x)
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
    constexpr int pow(BaseType base, ExpType exponent){
        if(exponent < 0)
            return 0; 
        else if(exponent == 0){
            return 1; 
        }
        else{
            for(int i = 1; i < exponent; i++){
                base *= base; 
            }
            return base;
        }
    }
    
}

#endif // UTILITY_H