#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <IPerceptron.h>
#include <Config.h>
#include <Utility.h>
#include <array>
#include <ActivationFunctionEnum.h>

namespace perceptron{
    
    template<int inputCount, ActivationFunctionEnum actFunctionEnum, int inputCountWithBias = inputCount + 1>
    class Perceptron : public IPerceptron {
    public:
        constexpr Perceptron(int bias = 0){
            static_assert(inputCount > 0, "Input count must be greater than 0.");
            static_assert(inputCount +1 == inputCountWithBias, "Please do not use this template parameter.");
    
            inputs.fill(0.0f);
            weights.fill(1.0f);
    
            weights[0] = bias;
        }
        constexpr ~Perceptron() = default; 
    
        constexpr float CalculateOutput() {
            float sum = 0.0f;
            for(int i = 0; i < inputCountWithBias; i++){
                sum += inputs[i] * weights[i];
            }
            return ActivationFunction(sum);
        }
    
        constexpr float ActivationFunction(float input){
            if constexpr (actFunctionEnum == ActivationFunctionEnum::Linear){
                return utility::af::Linear(input);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::Sigmoid) {
                return utility::af::Sigmoid(input);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::Tanh) {
                return utility::af::Tanh(input);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::ReLU) {
                return utility::af::Relu(input);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::LeakyReLU) {
                return utility::af::LeakyRelu(input);
            }
            return input;
        }
        
    private: 
        std::array<float, inputCountWithBias> weights; //weights for each input and bias(+1)
        std::array<float, inputCountWithBias> inputs; //inputs for each input and bias(+1)
        float output;
    };
}


#endif