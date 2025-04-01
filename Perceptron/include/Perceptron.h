#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "IPerceptron.h"
#include <vector>
#include <utility>
#include <Config.h>
#include <array>
#include <type_traits>
#include <ActivationFunctionEnum.h>

namespace perceptron{
    
    template<int inputCount, ActivationFunctionEnum actFunctionEnum, int inputCountWithBias = inputCount + 1>
    class Perceptron : public IPerceptron {
    public:
        constexpr Perceptron(int bias = 0){
            static_assert(inputCount > 0, "Input count must be greater than 0.");
            static_assert(inputCount +1 == inputCountWithBias, "Please do not use this template parameter.");
    
            inputs.fill(1.0f);
            weights.fill(1.0f);
    
            weights[0] = bias;
        }
        constexpr ~Perceptron() = default; 
    
        constexpr float CalculateOutput(float input) {
            float sum = 0.0f;
            for(int i = 0; i < inputCountWithBias; i++){
                sum += inputs[i] * weights[i];
            }
            return 0;
        }
    
        constexpr float ActivationFunction(float input){
            if constexpr (actFunctionEnum == ActivationFunctionEnum::Linear){
                return input;
            }
            else{
                return input +1;
            }
        }
    private: 
        std::array<float, inputCountWithBias> weights; //weights for each input and bias(+1)
        std::array<float, inputCountWithBias> inputs; //inputs for each input and bias(+1)
        float output;
    };
}


#endif