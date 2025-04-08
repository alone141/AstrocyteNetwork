#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <IPerceptron.h>
#include <Config.h>
#include <Utility.h>
#include <array>
#include <ActivationFunctionEnum.h>
#include <iostream>
namespace perceptron{
    
    template<int inputCount, ActivationFunctionEnum actFunctionEnum, int inputCountWithBias = inputCount + 1>
    class Perceptron : public IPerceptron {
        static_assert(inputCount > 0, "Input count must be greater than 0.");
        static_assert(inputCount +1 == inputCountWithBias, "Please do not use this template parameter.");
        public:
        constexpr Perceptron(int randomSeed) : output(0.0f) {
            for (int i=0; i < inputCountWithBias; i++)
            {
                weights[i] = utility::random::GetRandomUniform(randomSeed*(i+1));
            }
            //bias is always 1.0f           
            weights[0] = 1;
            inputs[0] = 1;
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
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::PseudoSigmoid) {
                return utility::af::PseudoSigmoid(input);
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

        //todo: bound check
        constexpr float GetWeight(unsigned int weightIndex) const override {
            if(weightIndex >= inputCountWithBias){
                return 0.0f; //or throw an exception idk yet
            }
            return weights[weightIndex];
        }
        constexpr void SetInput(unsigned int inputIndex, float inputValue) override {
            if(inputIndex >= inputCountWithBias){
                return;
            }
            inputs[inputIndex] = inputValue;
        }
        constexpr void SetWeight(unsigned int weightIndex, float weightValue) override {
            if(weightIndex >= inputCountWithBias){
                return;
            }
            weights[weightIndex] = weightValue;
        }

        //doesnt do what its supposed to do yet
        constexpr void ResetWeights() override {
            weights.fill(1.0f);
        }
        void Print() override {
            std::cout << "Weights: {  ";
            for (const auto& weight : weights) {
                std::cout << weight << ", ";
            }
            std::cout << "}" << std::endl;
        
            std::cout << "Inputs: { ";
            for (const auto& input : inputs) {
                std::cout << input << ", ";
            }
            std::cout << "}" << std::endl;
        
            std::cout << "Output: " << output << std::endl << std::endl;
        }
    private: 
        std::array<float, inputCountWithBias> weights; //weights for each input and bias(+1)
        std::array<float, inputCountWithBias> inputs; //inputs for each input and bias(+1)
        float output;
    };
}


#endif