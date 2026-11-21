#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <IPerceptron.h>
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
        constexpr explicit Perceptron(int randomSeed) : seed(randomSeed) {
            //the bias INPUT is always 1.0f, its WEIGHT (weights[0]) is random and learned like any other weight
            inputs[0] = 1.0f;
            ResetWeights();
        }
        constexpr ~Perceptron() = default; 

        constexpr float CalculateOutput() override {
            float sum = 0.0f;
            for(std::size_t i = 0; i < inputCountWithBias; i++){
                //custom inputs starts at index 1
                //index 0 is reserved for bias
                sum += inputs[i] * weights[i];
            }
            output = ActivationFunction(sum);
            return output;
        }
    
        constexpr float ActivationFunction(const float input) const override {
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

        // Backpropagation needs the slope of the activation function. It must match ActivationFunction,
        // so both are selected from the same enum here instead of hard-coding the sigmoid derivative.
        constexpr float ActivationDerivative() const override {
            if constexpr (actFunctionEnum == ActivationFunctionEnum::Linear){
                return utility::af::derivative::Linear(output);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::Sigmoid) {
                return utility::af::derivative::Sigmoid(output);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::PseudoSigmoid) {
                return utility::af::derivative::PseudoSigmoid(output);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::Tanh) {
                return utility::af::derivative::Tanh(output);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::ReLU) {
                return utility::af::derivative::Relu(output);
            }
            else if constexpr (actFunctionEnum == ActivationFunctionEnum::LeakyReLU) {
                return utility::af::derivative::LeakyRelu(output);
            }
            return 1.0f;
        }


        constexpr float GetOutput() const override {
            return output;
        }

        constexpr float GetWeight(const unsigned int weightIndex) const override {
            if(weightIndex >= inputCountWithBias){
                return 0.0f; //or throw an exception idk yet
            }
            return weights[weightIndex];
        }
        constexpr void SetInput(const unsigned int inputIndex, float inputValue) override {
            if(inputIndex >= inputCountWithBias){
                return;
            }
            inputs[inputIndex] = inputValue;
        }
        constexpr void SetWeight(const unsigned int weightIndex, float weightValue) override {
            if(weightIndex >= inputCountWithBias){
                return;
            }
            weights[weightIndex] = weightValue;
        }

        // Draws this perceptron's initial random weights. Each perceptron uses its own PCG stream (its seed),
        // so the weights are different for every perceptron but identical on every build and every reset.
        // Weights must never all be equal: identical neurons get identical gradients and can never become different.
        constexpr void ResetWeights() override {
            utility::random::PCG rng(config::RANDOM_SEED, static_cast<std::uint64_t>(seed));
            for (auto& weight : weights) {
                weight = utility::random::GetRandomUniform(rng);
            }
        }

        void Print() const override {
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
        int seed; //random stream of this perceptron, used by ResetWeights
        std::array<float, inputCountWithBias> weights{}; //weights for each input and bias(+1)
        std::array<float, inputCountWithBias> inputs{}; //inputs for each input and bias(+1)
        float output = 0.0f;
    };
}


#endif