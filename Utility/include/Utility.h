#ifndef UTILITY_H
#define UTILITY_H

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

    constexpr int pow(float base, int exponent){
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
    constexpr float pow(auto base, auto exponent) = delete;
    
}

#endif // UTILITY_H