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
    
            for (int i = 0; i < inputCountWithBias; i++)
            {
                weights[i] = utility::random::GetRandomUniform(i);
            }
            
            
            inputs.fill(0.0f);
/*             weights.fill(0.6f); */
            
            weights[0] = bias;
            inputs[0] = bias;
        }
        constexpr ~Perceptron() = default; 
    
        constexpr float CalculateOutput() override {
            float sum = 0.0f;
            for(int i = 0; i < inputCountWithBias; i++){
                //custom inputs starts at index 1
                //index 0 is reserved for bias
                sum += inputs[i] * weights[i];
            }
            output = ActivationFunction(sum);
            return output;
        }
    
        constexpr float ActivationFunction(float input) override {
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
        
        constexpr float GetOutput() const override {
            return output;
        }
        constexpr void SetInput(unsigned int inputIndex, float inputValue) override {
            inputs[inputIndex] = inputValue;
        }

        //doesnt do what its supposed to do yet
        constexpr void ResetWeights() override {
            weights.fill(1.0f);
        }
    private: 
        std::array<float, inputCountWithBias> weights; //weights for each input and bias(+1)
        std::array<float, inputCountWithBias> inputs; //inputs for each input and bias(+1)
        float output;
    };
}


#endif